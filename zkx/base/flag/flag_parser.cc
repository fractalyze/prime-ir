// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.console file.

/* Copyright 2025 The ZKX Authors.

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

#include "zkx/base/flag/flag_parser.h"

#include <iostream>
#include <limits>
#include <tuple>

#include "xla/tsl/platform/errors.h"

namespace zkx::base {

namespace {

constexpr int kDefaultHelpStart = 20;

bool ContainsOnlyAlpha(std::string_view text) {
  return std::all_of(text.begin(), text.end(),
                     [](char c) { return absl::ascii_isalpha(c); });
}

void AppendActiveSubParser(
    std::ostream& ss, FlagParserBase& parser,
    const std::vector<std::unique_ptr<FlagBase>>** flags) {
  bool has_subparser = false;

  if (parser.flags()->size() > 0) {
    has_subparser = (*parser.flags())[0]->IsSubParser();
  }

  if (has_subparser) {
    for (const auto& flag : *parser.flags()) {
      if (flag->IsSubParser()) {
        if (flag->is_set()) {
          ss << " " << flag->name();
          AppendActiveSubParser(ss, *flag->ToSubParser(), flags);
          return;
        }
      } else {
        break;
      }
    }
  }
  *flags = parser.flags();
}

struct Costs {
  explicit Costs(size_t size) : size(size) {
    if (size > 256) {
      costs = new size_t[size];
    } else {
      costs = costs_on_stack;
    }
    Init();
  }

  void Init() {
    for (size_t i = 0; i < size; ++i) {
      costs[i] = i;
    }
  }

  ~Costs() {
    if (size > 256) {
      delete[] costs;
    }
  }

  size_t& operator[](size_t i) { return costs[i]; }

  size_t size;
  size_t* costs = nullptr;
  size_t costs_on_stack[256];
};

// This was taken and modified from
// https://rosettacode.org/wiki/Levenshtein_distance#C.2B.2B
size_t GetLevenshteinDistance(std::string_view s1, std::string_view s2) {
  size_t m = s1.size();
  size_t n = s2.size();

  if (m == 0) return n;
  if (n == 0) return m;

  Costs costs(n + 1);

  size_t i = 0;
  for (; i < m; ++i) {
    costs[0] = i + 1;
    size_t corner = i;
    size_t j = 0;
    char c = s1[i];
    for (; j < n; ++j) {
      size_t upper = costs[j + 1];
      if (c == s2[j]) {
        costs[j + 1] = corner;
      } else {
        size_t t = upper < corner ? upper : corner;
        costs[j + 1] = (costs[j] < t ? costs[j] : t) + 1;
      }

      corner = upper;
    }
  }

  return costs[n];
}

}  // namespace

FlagParserBase::Context::Context(FlagParser* parser, int current_idx, int argc,
                                 char** argv)
    : parser(parser), current_idx(current_idx), argc(argc), argv(argv) {}

FlagParserBase::Context::~Context() = default;

std::string_view FlagParserBase::Context::current() const {
  return argv[current_idx];
}

bool FlagParserBase::Context::ConsumeEqualOrProceed(std::string_view* arg) {
  if (absl::ConsumePrefix(arg, "=")) return true;
  if (!arg->empty()) return false;
  Proceed();
  *arg = current();
  return true;
}

void FlagParserBase::Context::Proceed() { current_idx++; }

bool FlagParserBase::Context::HasArg() const { return current_idx < argc; }

void FlagParserBase::Context::FillUnknownArgs() const {
  if (!unknown_argc) return;
  size_t new_argc = 1;
  for (char* unknown_arg : unknown_argv) {
    argv[new_argc++] = unknown_arg;
  }
  *unknown_argc = new_argc;
}

FlagParserBase::FlagParserBase() = default;

FlagParserBase::~FlagParserBase() = default;

FlagBase& FlagParserBase::AddFlag(std::unique_ptr<FlagBase> flag_base) {
  flags_.push_back(std::move(flag_base));
  return *flags_.back().get();
}

SubParser& FlagParserBase::AddSubParser() {
  std::unique_ptr<SubParser> sub_parser(new SubParser());
  flags_.push_back(std::move(sub_parser));
  return *flags_.back()->ToSubParser();
}

absl::Status FlagParserBase::ValidateInternally() const {
  bool is_positional = true;
  bool has_subparser = false;
  for (auto& flag : flags_) {
    if (!(flag->is_positional() || flag->is_optional())) {
      return absl::InvalidArgumentError(
          "Flag should be positional or optional.");
    }

    if (flag->is_positional() && flag->is_optional()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "\"$0\" is both positional and optional, please choose either one of "
          "them.",
          flag->name()));
    }

    if (flag->IsSubParser()) {
      if (flag->is_optional()) {
        return absl::InvalidArgumentError(absl::Substitute(
            "Subparser \"$0\" should be positional.", flag->display_name()));
      }

      has_subparser = true;
      TF_RETURN_IF_ERROR(flag->ToSubParser()->ValidateInternally());
    } else {
      if (flag->is_positional()) {
        if (!is_positional) {
          return absl::InvalidArgumentError(absl::Substitute(
              "\"$0\" should be before any optional arguments.",
              flag->display_name()));
        }
      } else {
        is_positional = false;
      }
    }

    if (!flag->NeedsValue()) {
      if (flag->is_positional()) {
        return absl::InvalidArgumentError(absl::Substitute(
            "\"$0\" can't parse a value, how about considering using "
            "set_short_name() or set_long_name()?",
            flag->name()));
      }
    }
  }

  bool should_not_be_subparser = false;
  if (has_subparser) {
    for (auto& flag : flags_) {
      if (flag->is_positional()) {
        if (!flag->IsSubParser()) {
          return absl::InvalidArgumentError(
              absl::Substitute("\"$0\" can't be positional if the parser has "
                               "subparser, how about considering using "
                               "set_short_name() or set_long_name()?",
                               flag->name()));
        } else if (should_not_be_subparser) {
          return absl::InvalidArgumentError(
              "SubParser should be at the very front.");
        }
      } else {
        should_not_be_subparser = true;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status FlagParserBase::Parse(Context& ctx) {
  int positional_parsed = 0;
  int positional_argument = std::count_if(
      flags_.begin(), flags_.end(), [](const std::unique_ptr<FlagBase>& flag) {
        return flag->is_positional();
      });
  bool has_subparser = false;
  if (positional_argument > 0) {
    has_subparser = flags_[0]->IsSubParser();
  }

  while (ctx.HasArg()) {
    std::string_view arg = ctx.current();
    if (arg == "--") {
      if (ctx.forward_argv) {
        ctx.forward_argv->reserve(ctx.argc - ctx.current_idx - 1);
        for (int i = ctx.current_idx + 1; i < ctx.argc; ++i) {
          ctx.forward_argv->emplace_back(ctx.argv[i]);
        }
      }
      return absl::OkStatus();
    }

    if (arg == "--help" || arg == "-h") {
      std::cerr << ctx.parser->help_message() << std::endl;
      return absl::AbortedError(absl::Substitute("Got \"$0\".", arg));
    }

    FlagBase* target_flag = nullptr;
    if (has_subparser) {
      for (int i = 0; i < positional_argument; ++i) {
        if (flags_[i]->name() == arg) {
          target_flag = flags_[i].get();
          break;
        }
      }
    }

    if (!has_subparser && positional_parsed < positional_argument) {
      int positional_idx = 0;
      for (auto& flag : flags_) {
        if (flag->is_optional()) {
          continue;
        } else {
          if (positional_idx == positional_parsed) {
            target_flag = flag.get();
            break;
          }

          positional_idx++;
        }
      }
    } else {
      std::string_view new_arg = arg;
      if (absl::ConsumePrefix(&new_arg, "-")) {
        if ((ContainsOnlyAlpha(new_arg))) {
          // Flags can be passed like '-ab' if '-a' and '-b' are valid short
          // names and don't need values. For this special case, we have to
          // treat differently.
          bool all_parsed =
              std::all_of(new_arg.begin(), new_arg.end(), [this](char c) {
                for (auto& flag : flags_) {
                  if (flag->is_positional()) continue;
                  if (flag->NeedsValue()) continue;
                  if (flag->short_name().length() == 2 &&
                      c == flag->short_name()[1]) {
                    absl::Status s = flag->ParseValue("");
                    std::ignore = s;
                    DCHECK(s.ok());
                    return true;
                  }
                }

                return false;
              });
          if (all_parsed) {
            ctx.Proceed();
            continue;
          }
        }
      }

      for (auto& flag : flags_) {
        if (flag->is_positional()) continue;

        if (!flag->ConsumeNamePrefix(*this, &arg)) continue;
        target_flag = flag.get();
        break;
      }
    }

    absl::Status parsed_status = absl::UnknownError("");
    if (target_flag) {
      if (target_flag->IsSubParser()) {
        target_flag->is_set_ = true;
        SubParser* sub_parser = target_flag->ToSubParser();
        if (sub_parser->is_set_) *sub_parser->is_set_ = true;
        ctx.Proceed();
        return target_flag->ToSubParser()->Parse(ctx);
      } else if (target_flag->is_positional()) {
        parsed_status = target_flag->ParseValue(arg);
        positional_parsed++;
      } else {
        if (target_flag->NeedsValue() && !ctx.ConsumeEqualOrProceed(&arg)) {
          return absl::InvalidArgumentError(
              absl::Substitute("Failed to parse \"$0\": (reason: empty value).",
                               target_flag->display_name()));
        }
        parsed_status = target_flag->ParseValue(arg);
      }
    } else {
      if (ctx.unknown_argc) {
        ctx.unknown_argv.push_back(ctx.argv[ctx.current_idx]);
        ctx.Proceed();
        if (ctx.HasArg() && !absl::StartsWith(ctx.current(), "-")) {
          ctx.unknown_argv.push_back(ctx.argv[ctx.current_idx]);
          ctx.Proceed();
        }
        continue;
      }
    }

    if (!parsed_status.ok()) {
      if (target_flag) {
        if (parsed_status.message().empty()) {
          return absl::UnknownError(
              absl::Substitute("Failed to parse \"$0\": (reason: unknown).",
                               target_flag->display_name()));
        } else {
          return absl::Status(
              parsed_status.code(),
              absl::Substitute("Failed to parse \"$0\": (reason: $1).",
                               target_flag->display_name(),
                               parsed_status.message()));
        }
      } else {
        absl::StatusOr<std::string_view> candidate_arg =
            FindTheMostSimilarFlag(arg);
        if (candidate_arg.ok()) {
          return absl::InvalidArgumentError(absl::Substitute(
              "Met unknown argument: \"$0\", maybe you mean \"$1\"?", arg,
              candidate_arg.value()));
        } else {
          return absl::InvalidArgumentError(
              absl::Substitute("Met unknown argument: \"$0\".", arg));
        }
      }
    }
    ctx.Proceed();
  }

  if (!has_subparser && positional_parsed < positional_argument) {
    for (auto& flag : flags_) {
      if (flag->is_optional()) continue;
      if (!flag->is_set()) {
        return absl::InvalidArgumentError(absl::Substitute(
            "\"$0\" is positional, but not set.", flag->name()));
      }
    }
  }

  for (auto& flag : flags_) {
    if (!flag->is_set()) {
      if (flag->is_required()) {
        return absl::InvalidArgumentError(absl::Substitute(
            "\"$0\" is required, but not set.", flag->display_name()));
      } else {
        TF_RETURN_IF_ERROR(flag->ParseValueFromEnvironment());
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string_view> FlagParserBase::FindTheMostSimilarFlag(
    std::string_view input) {
  size_t threshold = (input.length() + 1) / 2;
  size_t min = std::numeric_limits<size_t>::max();
  std::string_view output;

  for (auto& flag : flags_) {
    if (!flag->short_name().empty()) {
      size_t dist = GetLevenshteinDistance(input, flag->short_name());
      if (dist <= threshold && dist < min) {
        output = flag->short_name();
        min = dist;
      }
    }

    if (!flag->long_name().empty()) {
      size_t dist = GetLevenshteinDistance(input, flag->long_name());
      if (dist <= threshold && dist < min) {
        output = flag->long_name();
        min = dist;
      }
    }

    if (flag->IsSubParser()) {
      size_t dist = GetLevenshteinDistance(input, flag->name());
      if (dist <= threshold && dist < min) {
        output = flag->name();
        min = dist;
      }
    }
  }

  if (min != std::numeric_limits<size_t>::max()) {
    return output;
  } else {
    return absl::NotFoundError("unable to find the most similar flag");
  }
}

FlagParser::FlagParser() = default;

FlagParser::~FlagParser() = default;

absl::Status FlagParser::Parse(int argc, char** argv) {
  return ParseWithForward(argc, argv, nullptr);
}

absl::Status FlagParser::ParseKnown(int* argc, char** argv) {
  return ParseKnownWithForward(argc, argv, nullptr);
}

absl::Status FlagParser::ParseWithForward(int argc, char** argv,
                                          std::vector<std::string>* forward) {
  TF_RETURN_IF_ERROR(PreParse(argc, argv));

  Context ctx(this, 1, argc, argv);
  ctx.forward_argv = forward;
  TF_RETURN_IF_ERROR(FlagParserBase::Parse(ctx));
  TF_RETURN_IF_ERROR(Validate());
  return absl::OkStatus();
}

absl::Status FlagParser::ParseKnownWithForward(
    int* argc, char** argv, std::vector<std::string>* forward) {
  TF_RETURN_IF_ERROR(PreParse(*argc, argv));

  Context ctx(this, 1, *argc, argv);
  ctx.unknown_argc = argc;
  ctx.forward_argv = forward;
  TF_RETURN_IF_ERROR(FlagParserBase::Parse(ctx));
  TF_RETURN_IF_ERROR(Validate());
  ctx.FillUnknownArgs();
  return absl::OkStatus();
}

absl::Status FlagParser::PreParse(int argc, char** argv) {
  TF_RETURN_IF_ERROR(ValidateInternally());

  if (program_path_.empty()) {
    DCHECK_GT(strlen(argv[0]), 0U);
    program_path_ = std::string(argv[0]);
  }

  return absl::OkStatus();
}

std::string FlagParser::help_message() {
  std::stringstream ss;
  ss << "Usage: " << std::endl << std::endl;
  ss << program_path_;
  const std::vector<std::unique_ptr<FlagBase>>* flags = nullptr;
  AppendActiveSubParser(ss, *this, &flags);

  // `flags` might be different from `flags_`.
  bool has_optional_flag = std::any_of(
      flags->begin(), flags->end(), [](const std::unique_ptr<FlagBase>& flag) {
        return flag->is_optional();
      });
  bool has_sub_parser = false;
  if (flags->size() > 0) {
    has_sub_parser = (*flags)[0]->IsSubParser();
  }
  if (has_sub_parser) {
    if (has_optional_flag) {
      ss << " [OPTIONS]";
    }
    ss << " COMMAND";
  }

  if (has_sub_parser) {
    ss << std::endl << std::endl;
    ss << "Commands:" << std::endl << std::endl;
    for (auto& flag : *flags) {
      if (flag->IsSubParser()) {
        ss << flag->display_help(kDefaultHelpStart) << std::endl;
      }
    }
  } else {
    bool has_positional_flag =
        std::any_of(flags->begin(), flags->end(),
                    [](const std::unique_ptr<FlagBase>& flag) {
                      return flag->is_positional();
                    });
    if (has_positional_flag) {
      for (auto& flag : *flags) {
        if (flag->is_positional()) {
          ss << " " << flag->name();
        }
      }
    }
    if (has_optional_flag) {
      ss << " [OPTIONS]";
    }
    ss << std::endl;

    if (has_positional_flag) {
      ss << std::endl << "Positional arguments:" << std::endl << std::endl;
      for (auto& flag : *flags) {
        if (flag->is_positional()) {
          ss << flag->display_help(kDefaultHelpStart) << std::endl;
        }
      }
    }
  }

  if (has_optional_flag) {
    ss << std::endl << "Optional arguments:" << std::endl << std::endl;
    for (auto& flag : *flags) {
      if (flag->is_optional()) {
        ss << flag->display_help(kDefaultHelpStart) << std::endl;
      }
    }
  }

  return ss.str();
}

SubParser::SubParser() = default;

SubParser::~SubParser() = default;

bool SubParser::IsSubParser() const { return true; }

bool SubParser::NeedsValue() const { return true; }

absl::Status SubParser::ParseValue(std::string_view arg) {
  return absl::InternalError("Should not be called");
}

}  // namespace zkx::base
