/* Copyright 2020 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/hlo/translate/mhlo_to_hlo/attribute_exporter.h"

#include "mlir/IR/BuiltinTypes.h"

#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/hlo/parser/hlo_parser.h"

namespace zkx {

// Convert replica group from MLIR encoding to HLO.
// See HloFunctionImporter::ConvertReplicaGroups for the MLIR encoding.
absl::StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input) {
  mlir::RankedTensorType type =
      mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!type || type.getRank() != 2 ||
      !type.getElementType().isInteger(/*width=*/64)) {
    return absl::InternalError(
        "Expected replica group to be a rank 2 tensor of i64");
  }
  // rank 0 is num_groups, rank 1 is group size.
  auto replica_group_values_it = input.getValues<uint64_t>().begin();
  std::vector<ReplicaGroup> replica_groups(type.getDimSize(0));
  for (ReplicaGroup& group : replica_groups) {
    for (int64_t element_idx = 0; element_idx < type.getDimSize(1);
         ++element_idx, ++replica_group_values_it) {
      // For replica group attribute, -1 indicates padding added by
      // HloFunctionImporter::ConvertReplicaGroups. This should always be at the
      // end and can be dropped when converting back to XLA HLO ReplicaGroups.
      if (*replica_group_values_it != -1) {
        group.add_replica_ids(*replica_group_values_it);
      }
    }
  }
  return replica_groups;
}

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    std::optional<mlir::DenseIntElementsAttr> optional_attr) {
  if (!optional_attr.has_value())
    return std::vector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optional_attr;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!type || type.getRank() != 2 || type.getShape()[1] != 2)
    return absl::InternalError(
        "Expected Nx2 attribute to be a tensor of shape Nx2");
  auto it = attr.getValues<int64_t>().begin();
  std::vector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto& item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}

std::optional<OpSharding> ConvertSharding(llvm::StringRef sharding) {
  OpSharding sharding_proto;
  if (sharding_proto.ParseFromString(sharding.str())) return sharding_proto;
  absl::StatusOr<HloSharding> sharding_cpp = ParseSharding(sharding.str());
  if (sharding_cpp.ok()) return sharding_cpp->ToProto();
  return std::nullopt;
}

std::optional<HloInputOutputAliasProto> ConvertInputOutputAlias(
    llvm::ArrayRef<mlir::Attribute> aliasing) {
  if (aliasing.empty()) return std::nullopt;

  HloInputOutputAliasProto input_output_alias_proto;
  for (auto attr : aliasing) {
    auto entry_attr = mlir::cast<mlir::DictionaryAttr>(attr);
    auto alias_attr = mlir::cast<mlir::DictionaryAttr>(entry_attr.get("alias"));
    mlir::ArrayRef<int64_t> output_index =
        mlir::cast<mlir::DenseI64ArrayAttr>(entry_attr.get("output_index"))
            .asArrayRef();
    mlir::ArrayRef<int64_t> parameter_index =
        mlir::cast<mlir::DenseI64ArrayAttr>(alias_attr.get("parameter_index"))
            .asArrayRef();
    HloInputOutputAliasProto::AliasEntryProto entry;
    entry.mutable_output_shape_index()->Add(output_index.begin(),
                                            output_index.end());
    entry.set_parameter_number(
        mlir::cast<mlir::IntegerAttr>(alias_attr.get("parameter_number"))
            .getInt());
    entry.mutable_parameter_shape_index()->Add(parameter_index.begin(),
                                               parameter_index.end());
    mlir::StringRef kind =
        mlir::cast<mlir::StringAttr>(alias_attr.get("kind")).getValue();
    if (kind == "may_alias")
      entry.set_kind(Kind::MAY_ALIAS);
    else if (kind == "must_alias")
      entry.set_kind(Kind::MUST_ALIAS);
    else
      entry.set_kind(Kind::UNDEFINED_ALIAS);
    input_output_alias_proto.add_entries()->Swap(&entry);
  }
  return input_output_alias_proto;
}

}  // namespace zkx
