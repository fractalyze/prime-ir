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

#ifndef ZKX_BASE_FLAG_FLAG_H_
#define ZKX_BASE_FLAG_FLAG_H_

#include <algorithm>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/strip.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest_prod.h"

#include "xla/tsl/platform/env.h"
#include "zkx/base/flag/flag_forward.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx::base {

bool IsValidFlagName(std::string_view text);

class FlagParserBase;
class FlagParser;
class SubParser;

template <typename T>
class FlagBaseBuilder {
 public:
  T& set_short_name(std::string_view short_name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = short_name;
    if (!absl::ConsumePrefix(&text, "-")) return *impl;
    if (!(text.length() == 1 && absl::ascii_isalpha(text[0]))) return *impl;

    impl->short_name_ = std::string(short_name);
    return *impl;
  }

  T& set_long_name(std::string_view long_name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = long_name;
    if (!absl::ConsumePrefix(&text, "--")) return *impl;
    if (!IsValidFlagName(text)) return *impl;

    impl->long_name_ = std::string(long_name);
    return *impl;
  }

  T& set_name(std::string_view name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = name;
    if (!IsValidFlagName(text)) return *impl;

    impl->name_ = std::string(name);
    return *impl;
  }

  T& set_help(std::string_view help) {
    T* impl = static_cast<T*>(this);
    impl->help_ = std::string(help);
    return *impl;
  }

  T& set_required() {
    T* impl = static_cast<T*>(this);
    impl->is_required_ = true;
    return *impl;
  }
};

// A FlagBase must define either `short_name_`, `long_name_`, or `name_`.
//
// - `short_name_`: Must be a single alphabet character prefixed with a single
//    dash (e.g., `-a`).
// - `long_name_`: Must start with an alphabet and contain only alphabets,
//    digits, or underscores, prefixed with two dashes (e.g., `--foo`).
// - `name_`: Must start with an alphabet and contain only alphabets, digits, or
//    underscores, without any prefix (e.g., `my_flag`).
//
// Notes:
// - `long_name_` and `short_name_` can be set together.
// - `name_` must not be used together with either `short_name_`
//    or `long_name_`.
// - Invalid examples: `--3a`, `--_ab` (names must start with an alphabet).

class FlagBase {
 public:
  FlagBase();
  FlagBase(const FlagBase& other) = delete;
  FlagBase& operator=(const FlagBase& other) = delete;
  virtual ~FlagBase();

  const std::string& short_name() const { return short_name_; }
  const std::string& long_name() const { return long_name_; }
  const std::string& name() const { return name_; }
  const std::string& help() const { return help_; }

  // Returns true if the flag was marked with required.
  bool is_required() const { return is_required_; }
  // Returns true `name_` was set.
  bool is_positional() const { return !name_.empty(); }
  // Returns true `short_name_` or `long_name_` was set.
  bool is_optional() const {
    return !short_name_.empty() || !long_name_.empty();
  }
  // Returns true if a value was set.
  bool is_set() const { return is_set_; }
  // Returns true if a flag is an instance of SubParser
  virtual bool IsSubParser() const { return false; }

  SubParser* ToSubParser() {
    DCHECK(IsSubParser());
    return reinterpret_cast<SubParser*>(this);
  }

  // Returns `name_` if it is positional.
  // Otherwise, it returns `long_name_` if it is not empty.
  // Returns `short_name_` if `long_name_` is empty.
  const std::string& display_name() const;
  std::string display_help(int help_start = 0) const;

 protected:
  template <typename T>
  friend class FlagBaseBuilder;
  friend class FlagParserBase;
  FRIEND_TEST(FlagParserTest, SubParserTest);

  bool ConsumeNamePrefix(FlagParserBase& parser, std::string_view* arg) const;

  // Returns true if underlying type of Flag<T>, in other words, T is bool.
  virtual bool NeedsValue() const = 0;
  virtual absl::Status ParseValue(std::string_view arg) = 0;
  virtual absl::Status ParseValueFromEnvironment() {
    return absl::UnimplementedError(
        "parsing value from environment is not implemented");
  }

  void reset() { is_set_ = false; }

  std::string short_name_;
  std::string long_name_;
  std::string name_;
  std::string help_;
  bool is_required_ = false;
  bool is_set_ = false;
};

template <typename T, typename value_type>
class FlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }
};

template <typename T>
class Flag : public FlagBase, public FlagBuilder<Flag<T>, T> {
 public:
  using value_type = T;
  using ParseValueCallback = std::function<absl::Status(std::string_view, T*)>;

  explicit Flag(T* value) : value_(value) {}
  Flag(T* value, ParseValueCallback parse_value_callback)
      : value_(value), parse_value_callback_(parse_value_callback) {}
  Flag(const Flag& other) = delete;
  Flag& operator=(const Flag& other) = delete;

 private:
  friend class FlagBuilder<Flag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);
  FRIEND_TEST(TimeDeltaFlagTest, ParseValue);

  const T* value() const { return value_; }

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  absl::Status ParseValue(std::string_view arg) override;
  absl::Status ParseValueFromEnvironment() override;

  T* value_ = nullptr;  // not owned
  ParseValueCallback parse_value_callback_;
  std::string env_name_;
};

template <typename T>
absl::Status Flag<T>::ParseValue(std::string_view arg) {
  if (parse_value_callback_) {
    absl::Status s = parse_value_callback_(arg, value_);
    if (s.ok()) {
      is_set_ = true;
    }
    return s;
  }

  absl::Status s = FlagValueTraits<T>::ParseValue(arg, value_);
  if (s.ok()) {
    is_set_ = true;
  }
  return s;
}

template <typename T>
absl::Status Flag<T>::ParseValueFromEnvironment() {
  if (!env_name_.empty()) {
    if (const char* value = std::getenv(env_name_.data())) {
      return ParseValue(value);
    }
  }
  return absl::OkStatus();
}

template <typename T, typename value_type>
class ChoicesFlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    DCHECK(impl->Contains(value));
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }
};

template <typename T>
class ChoicesFlag : public FlagBase,
                    public ChoicesFlagBuilder<ChoicesFlag<T>, T> {
 public:
  using value_type = T;

  ChoicesFlag(T* value, const std::vector<T>& choices)
      : value_(value), choices_(choices) {}
  ChoicesFlag(T* value, std::vector<T>&& choices)
      : value_(value), choices_(std::move(choices)) {}
  ChoicesFlag(const ChoicesFlag& other) = delete;
  ChoicesFlag& operator=(const ChoicesFlag& other) = delete;

 private:
  friend class ChoicesFlagBuilder<ChoicesFlag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  const T* value() const { return value_; }

  bool Contains(const T& value) {
    return std::find(choices_.begin(), choices_.end(), value) != choices_.end();
  }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  absl::Status ParseValue(std::string_view arg) override;
  absl::Status ParseValueFromEnvironment() override;

  T* value_ = nullptr;
  std::vector<T> choices_;
  std::string env_name_;
};

template <typename T>
absl::Status ChoicesFlag<T>::ParseValue(std::string_view arg) {
  T value;
  absl::Status s = FlagValueTraits<T>::ParseValue(arg, &value);
  if (s.ok()) {
    if (Contains(value)) {
      *value_ = std::move(value);
      is_set_ = true;
      return s;
    } else {
      return absl::NotFoundError(absl::Substitute("$0 is not in choices", arg));
    }
  }
  return s;
}

template <typename T>
absl::Status ChoicesFlag<T>::ParseValueFromEnvironment() {
  if (!env_name_.empty()) {
    if (const char* value = std::getenv(env_name_.data())) {
      return ParseValue(value);
    }
  }
  return absl::OkStatus();
}

template <typename T, typename value_type>
class RangeFlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    DCHECK(impl->Contains(value));
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }

  T& set_greater_than_or_equal_to(bool greater_than_or_equal_to) {
    T* impl = static_cast<T*>(this);
    impl->greater_than_or_equal_to_ = greater_than_or_equal_to;
    return *impl;
  }

  T& set_less_than_or_equal_to(bool less_than_or_equal_to) {
    T* impl = static_cast<T*>(this);
    impl->less_than_or_equal_to_ = less_than_or_equal_to;
    return *impl;
  }
};

template <typename T>
class RangeFlag : public FlagBase, public RangeFlagBuilder<RangeFlag<T>, T> {
 public:
  using value_type = T;

  RangeFlag(T* value, const T& start, const T& end)
      : value_(value), start_(start), end_(end) {
    DCHECK_GE(end, start);
  }
  RangeFlag(const RangeFlag& other) = delete;
  RangeFlag& operator=(const RangeFlag& other) = delete;

 private:
  friend class RangeFlagBuilder<RangeFlag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  const T* value() const { return value_; }

  bool Contains(const T& value) {
    if (greater_than_or_equal_to_) {
      if (value < start_) return false;
    } else {
      if (value <= start_) return false;
    }
    if (less_than_or_equal_to_) {
      return value <= end_;
    } else {
      return value < end_;
    }
  }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  absl::Status ParseValue(std::string_view arg) override;
  absl::Status ParseValueFromEnvironment() override;

  T* value_ = nullptr;  // not owned
  T start_;
  T end_;
  bool greater_than_or_equal_to_ = false;
  bool less_than_or_equal_to_ = false;
  std::string env_name_;
};

template <typename T>
absl::Status RangeFlag<T>::ParseValue(std::string_view arg) {
  T value;
  absl::Status s = FlagValueTraits<T>::ParseValue(arg, &value);
  if (Contains(value)) {
    *value_ = std::move(value);
    is_set_ = true;
    return s;
  } else {
    return absl::OutOfRangeError(absl::Substitute("$0 is out of range", arg));
  }
  return s;
}

template <typename T>
absl::Status RangeFlag<T>::ParseValueFromEnvironment() {
  if (!env_name_.empty()) {
    if (const char* value = std::getenv(env_name_.data())) {
      return ParseValue(value);
    }
  }
  return absl::OkStatus();
}

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_FLAG_H_
