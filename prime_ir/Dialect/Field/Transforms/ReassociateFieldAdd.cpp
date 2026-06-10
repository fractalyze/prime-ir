/* Copyright 2026 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "prime_ir/Dialect/Field/Transforms/ReassociateFieldAdd.h"

#include <limits>
#include <queue>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_REASSOCIATEFIELDADD
#include "prime_ir/Dialect/Field/Transforms/ReassociateFieldAdd.h.inc"

namespace {

// Marks a value whose depth computation is in progress on the explicit DFS
// stack. Encountering it as an operand means a def-use cycle (possible only
// in graph regions); the cycle edge is treated as depth 0.
constexpr unsigned kVisiting = std::numeric_limits<unsigned>::max();

// A field.add whose operands and result all share one type. Mixed-type adds
// (extension field + base field) change the element type along the chain,
// so they act as chain boundaries.
bool isUniformAdd(AddOp add) {
  if (!add)
    return false;
  Type type = add.getOutput().getType();
  return add.getLhs().getType() == type && add.getRhs().getType() == type;
}

// SSA depth of the DAG defining `v`: block arguments and zero-operand ops
// (constants) have depth 0, an op result is 1 + the maximum operand depth.
// This is the rank used for tree placement — a deep operand is "late" and
// should fold into the tree last. Iterative so emitter-generated chains
// thousands of ops deep cannot overflow the C++ stack.
unsigned getDepth(Value v, DenseMap<Value, unsigned> &memo) {
  SmallVector<Value> stack{v};
  while (!stack.empty()) {
    Value cur = stack.back();
    auto it = memo.find(cur);
    if (it != memo.end() && it->second != kVisiting) {
      stack.pop_back();
      continue;
    }
    Operation *def = cur.getDefiningOp();
    if (!def || def->getNumOperands() == 0) {
      memo[cur] = 0;
      stack.pop_back();
      continue;
    }
    bool operandsReady = true;
    unsigned maxOperandDepth = 0;
    for (Value operand : def->getOperands()) {
      auto operandIt = memo.find(operand);
      if (operandIt == memo.end()) {
        stack.push_back(operand);
        operandsReady = false;
      } else if (operandIt->second != kVisiting) {
        maxOperandDepth = std::max(maxOperandDepth, operandIt->second);
      }
    }
    if (operandsReady) {
      memo[cur] = maxOperandDepth + 1;
      stack.pop_back();
    } else {
      memo[cur] = kVisiting;
    }
  }
  return memo.lookup(v);
}

// A tree node during reconstruction: `rank` is the depth at which the value
// is ready, `seq` makes heap ordering deterministic among equal ranks.
struct Node {
  Value value;
  unsigned rank;
  unsigned seq;
};

struct NodeGreater {
  bool operator()(const Node &a, const Node &b) const {
    return a.rank > b.rank || (a.rank == b.rank && a.seq > b.seq);
  }
};

// Rewrites the maximal add-tree rooted at `root` into a rank-ordered
// balanced tree. Returns false if the tree is too small to profit.
// `depthMemo` is shared across chains so common upstream DAGs are ranked
// once per function.
bool reassociateChain(AddOp root, IRRewriter &rewriter,
                      DenseMap<Value, unsigned> &depthMemo) {
  // Collect the leaves of the maximal tree of single-use, same-block,
  // uniform adds under `root`. `interior` is pre-order (parents first).
  SmallVector<Value> leaves;
  SmallVector<Operation *> interior{root};
  SmallVector<Value> worklist{root.getRhs(), root.getLhs()};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto add = dyn_cast_or_null<AddOp>(v.getDefiningOp());
    if (isUniformAdd(add) && v.hasOneUse() &&
        add->getBlock() == root->getBlock()) {
      interior.push_back(add);
      worklist.push_back(add.getRhs());
      worklist.push_back(add.getLhs());
    } else {
      leaves.push_back(v);
    }
  }
  // A 2-leaf "chain" is a single add; nothing to rebalance.
  if (leaves.size() < 3)
    return false;

  // Rank every leaf, then rebuild Huffman-style: repeatedly combine the two
  // lowest-rank nodes, the combined sum becoming ready one level later.
  // Equal-rank leaves pair up into a balanced tree; a late operand (high
  // rank) is only combined once the rest of the sum has caught up to it,
  // which keeps it off the critical path.
  std::priority_queue<Node, std::vector<Node>, NodeGreater> heap;
  unsigned seq = 0;
  for (Value leaf : leaves) {
    heap.push({leaf, getDepth(leaf, depthMemo), seq++});
  }
  rewriter.setInsertionPoint(root);
  while (heap.size() > 1) {
    Node a = heap.top();
    heap.pop();
    Node b = heap.top();
    heap.pop();
    auto add = AddOp::create(rewriter, root.getLoc(), a.value, b.value);
    heap.push({add.getOutput(), std::max(a.rank, b.rank) + 1, seq++});
  }
  // Drop memo entries for the values about to be erased: a later allocation
  // could reuse their storage, and a stale hit would return a bogus depth.
  for (Operation *op : interior)
    depthMemo.erase(op->getResult(0));
  rewriter.replaceOp(root, heap.top().value);
  // The old interior adds are now dead; pre-order erases users before defs.
  for (Operation *op : ArrayRef(interior).drop_front())
    rewriter.eraseOp(op);
  return true;
}

} // namespace

struct ReassociateFieldAdd
    : impl::ReassociateFieldAddBase<ReassociateFieldAdd> {
  using ReassociateFieldAddBase::ReassociateFieldAddBase;

  void runOnOperation() override {
    // Collect chain roots first: rewriting invalidates interior ops, and a
    // walk must not visit erased ops.
    SmallVector<AddOp> roots;
    getOperation()->walk([&](AddOp op) {
      if (!isUniformAdd(op))
        return;
      // An add consumed as the single use of a uniform add in the same
      // block is interior to that user's tree, not a root.
      if (op.getOutput().hasOneUse()) {
        auto user = dyn_cast<AddOp>(*op.getOutput().getUsers().begin());
        if (isUniformAdd(user) && user->getBlock() == op->getBlock())
          return;
      }
      roots.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    DenseMap<Value, unsigned> depthMemo;
    for (AddOp root : roots) {
      if (reassociateChain(root, rewriter, depthMemo))
        ++numReassociated;
    }
  }
};

} // namespace mlir::prime_ir::field
