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

#include "zkx/base/console/table_writer.h"

#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "absl/log/log.h"
#include "absl/strings/strip.h"

#include "xla/tsl/platform/platform.h"
#if defined(PLATFORM_POSIX)
#include <sys/ioctl.h>
#endif

namespace zkx::base {

TableWriter::TableWriter() = default;

TableWriter::TableWriter(const TableWriter& other) = default;

TableWriter& TableWriter::operator=(const TableWriter& other) = default;

TableWriter::TableWriter(TableWriter&& other) noexcept = default;

TableWriter& TableWriter::operator=(TableWriter&& other) noexcept = default;

TableWriter::~TableWriter() = default;

void TableWriter::SetElement(size_t row, size_t col, std::string_view element) {
  if (elements_.size() <= row) {
    elements_.resize(row + 1);
  }
  if (elements_[row].size() <= col) {
    elements_[row].resize(titles_.size());
  }

  if (strip_mode_ == StripMode::kBothAsciiWhitespace) {
    elements_[row][col] = std::string(absl::StripAsciiWhitespace(element));
  } else if (strip_mode_ == StripMode::kTrailingAsciiWhitespace) {
    elements_[row][col] =
        std::string(absl::StripTrailingAsciiWhitespace(element));
  } else {
    elements_[row][col] =
        std::string(absl::StripLeadingAsciiWhitespace(element));
  }

  if (column_widths_[col].length == Length::kAuto) {
    column_widths_[col].width =
        std::max(column_widths_[col].width, elements_[row][col].length());
  }
}

std::string TableWriter::ToString() const {
  std::stringstream ss;
  AppendTable(ss);
  return ss.str();
}

void TableWriter::Print(bool with_new_line) const {
  AppendTable(std::cout);
  if (with_new_line) {
    std::cout << std::endl;
  }
}

void TableWriter::AppendTable(std::ostream& os) const {
  uint16_t max_width = std::numeric_limits<uint16_t>::max();
  if (fit_to_terminal_width_) {
#if defined(PLATFORM_POSIX)
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
      max_width = w.ws_col;
    } else {
      PLOG(ERROR) << "Failed to get terminal width";
    }
#else
    LOG(WARNING) << "Not implemented for non-posix platforms";
#endif
  }

  const std::vector<ColumnWidth>* final_column_widths = &column_widths_;
  std::vector<ColumnWidth> column_widths_fit_to_window;

  size_t num_cols_to_print = column_widths_.size();
  size_t width = 0;
  for (size_t col = 0; col < column_widths_.size(); ++col) {
    width += column_widths_[col].width;
    if (width > max_width) {
      column_widths_fit_to_window = column_widths_;
      size_t overflow = width - max_width;
      if (col == 0) {
        column_widths_fit_to_window[0].width = max_width;
        num_cols_to_print = 1;
      } else {
        if (overflow >= column_widths_fit_to_window[col].width) {
          // This column cannot be displayed, so we only show previous columns.
          num_cols_to_print = col;
        } else {
          column_widths_fit_to_window[col].width -= overflow;
          num_cols_to_print = col + 1;
        }
      }
      final_column_widths = &column_widths_fit_to_window;
      break;
    }
    if (space_ > 0) {
      width += space_;
    }
  }
  absl::Span<const std::string> titles(titles_);
  AppendRow(os, titles.subspan(0, num_cols_to_print), final_column_widths,
            header_align_);
  if (elements_.size() > 0) os << std::endl;

  for (size_t row = 0; row < elements_.size(); ++row) {
    absl::Span<const std::string> cols(elements_[row]);
    AppendRow(os, cols.subspan(0, num_cols_to_print), final_column_widths,
              body_align_);
    if (row != elements_.size() - 1) os << std::endl;
  }
}

void TableWriter::AppendRow(std::ostream& os,
                            absl::Span<const std::string> contents,
                            const std::vector<ColumnWidth>* column_widths,
                            Align align) const {
  if (align == Align::kRight) {
    os << std::right;
  } else {
    os << std::left;
  }

  for (size_t col = 0; col < contents.size(); ++col) {
    size_t column_width = (*column_widths)[col].width;
    std::string_view content = contents[col];
    if (column_width >= content.length()) {
      if (align == Align::kCenter) {
        size_t left_padding = (column_width - content.length()) / 2;
        size_t right_padding = column_width - left_padding - content.length();
        os << std::string(left_padding, ' ') << content
           << std::string(right_padding, ' ');
      } else {
        os << std::setw(column_width) << content;
      }
    } else {
      os << content.substr(0, column_width);
    }
    if (space_ > 0 && col != contents.size() - 1) {
      os << std::string(space_, ' ');
    }
  }
}

TableWriterBuilder::TableWriterBuilder() = default;

TableWriterBuilder::~TableWriterBuilder() = default;

TableWriterBuilder& TableWriterBuilder::AlignHeaderLeft() {
  writer_.header_align_ = TableWriter::Align::kLeft;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AlignHeaderRight() {
  writer_.header_align_ = TableWriter::Align::kRight;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AlignHeaderCenter() {
  writer_.header_align_ = TableWriter::Align::kCenter;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AlignBodyLeft() {
  writer_.body_align_ = TableWriter::Align::kLeft;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AlignBodyRight() {
  writer_.body_align_ = TableWriter::Align::kRight;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AlignBodyCenter() {
  writer_.body_align_ = TableWriter::Align::kCenter;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AddSpace(size_t space) {
  writer_.space_ = space;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AddColumn(std::string_view title) {
  writer_.titles_.emplace_back(absl::StripAsciiWhitespace(title));
  writer_.column_widths_.push_back(
      {TableWriter::Length::kAuto, writer_.titles_.back().length()});
  return *this;
}

TableWriterBuilder& TableWriterBuilder::AddColumn(std::string_view title,
                                                  size_t width) {
  writer_.titles_.emplace_back(absl::StripAsciiWhitespace(title));
  writer_.column_widths_.push_back({TableWriter::Length::kFixed, width});
  return *this;
}

TableWriterBuilder& TableWriterBuilder::FitToTerminalWidth() {
  writer_.fit_to_terminal_width_ = true;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::StripBothAsciiWhitespace() {
  writer_.strip_mode_ = TableWriter::StripMode::kBothAsciiWhitespace;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::StripTrailingAsciiWhitespace() {
  writer_.strip_mode_ = TableWriter::StripMode::kTrailingAsciiWhitespace;
  return *this;
}

TableWriterBuilder& TableWriterBuilder::StripLeadingAsciiWhitespace() {
  writer_.strip_mode_ = TableWriter::StripMode::kLeadingAsciiWhitespace;
  return *this;
}

TableWriter TableWriterBuilder::Build() const { return writer_; }

}  // namespace zkx::base
