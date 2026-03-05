/* Copyright 2025 The OpenXLA Authors.
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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"         // IWYU pragma: keep
#include "nanobind/stl/string.h"       // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"       // IWYU pragma: keep

#include "xla/tsl/profiler/lib/traceme.h"
#include "xla/tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "xla/tsl/profiler/protobuf/xplane.pb.h"
#include "zkx/pjrt/exceptions.h"
#include "zkx/pjrt/status_casters.h"
#include "zkx/python/aggregate_profile.h"
#include "zkx/python/profiler/profile_data_lib.h"
#include "zkx/python/profiler_utils.h"
#include "zkx/python/xplane_to_profile_instructions.h"

namespace zkx {

namespace nb = nanobind;

namespace {

// Wraps TraceMe with an interface that takes python types.
class TraceMeWrapper {
 public:
  // nb::str and nb::kwargs are taken by const reference to avoid
  // python reference-counting overhead.
  TraceMeWrapper(const nb::str& name, const nb::kwargs& kwargs)
      : traceme_(
            [&]() {
              std::string name_and_metadata = nb::cast<std::string>(name);
              if (kwargs.size() > 0) {
                AppendMetadata(&name_and_metadata, kwargs);
              }
              return name_and_metadata;
            },
            /*level=*/1) {}

  // nb::kwargs is taken by const reference to avoid python
  // reference-counting overhead.
  void SetMetadata(const nb::kwargs& kwargs) {
    if (kwargs.size() > 0) {
      traceme_.AppendMetadata([&]() {
        std::string metadata;
        AppendMetadata(&metadata, kwargs);
        return metadata;
      });
    }
  }

  void Stop() { traceme_.Stop(); }

  static bool IsEnabled() { return tsl::profiler::TraceMe::Active(); }

 private:
  // Converts kwargs to strings and appends them to name encoded as TraceMe
  // metadata.
  static void AppendMetadata(std::string* name, const nb::kwargs& kwargs) {
    name->push_back('#');
    for (const auto& kv : kwargs) {
      absl::StrAppend(name, nb::cast<std::string_view>(kv.first), "=",
                      EncodePyObject(kv.second), ",");
    }
    name->back() = '#';
  }

  static std::string EncodePyObject(nb::handle handle) {
    if (nb::isinstance<nb::bool_>(handle)) {
      return nb::cast<bool>(handle) ? "1" : "0";
    }
    return nb::cast<std::string>(nb::str(handle));
  }

  tsl::profiler::TraceMe traceme_;
};

// Stub for ProfilerSessionWrapper. Full profiler session support requires
// tsl::ProfilerSession which is not yet ported to ZKX.
// TODO(chokobole): Port tsl/profiler/lib/profiler_session and
// tsl/profiler/rpc/ to enable full profiler session support.
struct ProfilerSessionWrapper {};

// Stub for ProfilerServer.
// TODO(chokobole): Port tsl/profiler/rpc/profiler_server to enable remote
// profiling support.
struct ProfilerServerWrapper {};

std::string GetFdoProfile(const std::string& xspace,
                          bool as_textproto = false) {
  tensorflow::profiler::XSpace xspace_proto;
  xspace_proto.ParseFromString(xspace);
  tensorflow::profiler::ProfiledInstructionsProto fdo_profile;
  zkx::ThrowIfError(zkx::ConvertXplaneToProfiledInstructionsProto(
      {xspace_proto}, &fdo_profile));
  if (as_textproto) {
    std::string textproto;
    if (google::protobuf::TextFormat::PrintToString(fdo_profile, &textproto)) {
      return textproto;
    }
    throw zkx::ZkxRuntimeError("Unable to serialize format to textproto");
  }
  return fdo_profile.SerializeAsString();
}

// Stub for tensorflow::ProfileOptions protobuf.
// Default values match XLA's DefaultPythonProfileOptions().
// TODO(chokobole): Replace with tensorflow::ProfileOptions when ported.
struct ProfileOptionsWrapper {
  bool include_dataset_ops = true;
  int32_t host_tracer_level = 2;
  int32_t python_tracer_level = 1;
  bool enable_hlo_proto = true;
  int64_t start_timestamp_ns = 0;
  int64_t duration_ms = 1000;
  bool raise_error_on_start_failure = true;
  std::string repository_path;
};

}  // namespace

NB_MODULE(_profiler, m) {
  nb::class_<ProfilerServerWrapper>(m, "ProfilerServer");
  m.def(
      "start_server",
      [](int port) -> ProfilerServerWrapper {
        // TODO(chokobole): Implement when tsl profiler RPC server is ported.
        throw zkx::ZkxRuntimeError(
            "Profiler server is not yet supported in ZKX.");
      },
      nb::arg("port"));
  m.def("register_plugin_profiler", [](nb::capsule c_api) -> void {
    if (std::string_view(c_api.name()) != "pjrt_c_api") {
      throw zkx::ZkxRuntimeError(
          "Argument to register_plugin_profiler was not a pjrt_c_api capsule.");
    }
    RegisterProfiler(c_api.data());
  });

  nb::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class
      .def("__init__",
           [](ProfilerSessionWrapper* wrapper) {
             new (wrapper) ProfilerSessionWrapper();
             // TODO(chokobole): Implement when tsl::ProfilerSession is ported.
             LOG(WARNING) << "ProfilerSession is not yet supported in ZKX. "
                          << "Profiling data will not be collected.";
           })
      .def("__init__",
           [](ProfilerSessionWrapper* wrapper,
              const ProfileOptionsWrapper& options) {
             new (wrapper) ProfilerSessionWrapper{};
             LOG(WARNING) << "ProfilerSession is not yet supported in ZKX. "
                          << "Profiling data will not be collected.";
           })
      .def("stop_and_export",
           [](ProfilerSessionWrapper* sess,
              const std::string& tensorboard_dir) -> void {
             LOG(WARNING) << "ProfilerSession.stop_and_export is a no-op in "
                             "ZKX.";
           })
      .def("stop",
           [](ProfilerSessionWrapper* sess) -> nb::bytes {
             LOG(WARNING) << "ProfilerSession.stop is a stub in ZKX. "
                          << "Returning empty XSpace.";
             tensorflow::profiler::XSpace empty_xspace;
             std::string s = empty_xspace.SerializeAsString();
             return nb::bytes(s.data(), s.size());
           })
      .def("stop_and_get_profile_data",
           [](ProfilerSessionWrapper* sess)
               -> tensorflow::profiler::python::ProfileData {
             LOG(WARNING) << "ProfilerSession.stop_and_get_profile_data is a "
                             "stub in ZKX.";
             auto xspace = std::make_shared<tensorflow::profiler::XSpace>();
             return tensorflow::profiler::python::ProfileData(xspace);
           })
      .def("export", [](ProfilerSessionWrapper* sess, nb::bytes xspace,
                        const std::string& tensorboard_dir) {
        LOG(WARNING) << "ProfilerSession.export is a no-op in ZKX.";
      });

  nb::class_<ProfileOptionsWrapper> profile_options_class(m, "ProfileOptions");
  profile_options_class
      .def("__init__",
           [](ProfileOptionsWrapper* options) {
             new (options) ProfileOptionsWrapper();
           })
      .def_rw("include_dataset_ops",
              &ProfileOptionsWrapper::include_dataset_ops)
      .def_rw("host_tracer_level", &ProfileOptionsWrapper::host_tracer_level)
      .def_rw("python_tracer_level",
              &ProfileOptionsWrapper::python_tracer_level)
      .def_rw("enable_hlo_proto", &ProfileOptionsWrapper::enable_hlo_proto)
      .def_rw("start_timestamp_ns", &ProfileOptionsWrapper::start_timestamp_ns)
      .def_rw("duration_ms", &ProfileOptionsWrapper::duration_ms)
      .def_rw("raise_error_on_start_failure",
              &ProfileOptionsWrapper::raise_error_on_start_failure)
      .def_rw("repository_path", &ProfileOptionsWrapper::repository_path);

  nb::class_<TraceMeWrapper> traceme_class(m, "TraceMe");
  traceme_class.def(nb::init<nb::str, nb::kwargs>())
      .def("__enter__", [](nb::object self) -> nb::object { return self; })
      .def(
          "__exit__",
          [](nb::object self, const nb::object& ex_type,
             const nb::object& ex_value,
             const nb::object& traceback) -> nb::object {
            nb::cast<TraceMeWrapper*>(self)->Stop();
            return nb::none();
          },
          nb::arg("ex_type").none(), nb::arg("ex_value").none(),
          nb::arg("traceback").none())
      .def("set_metadata", &TraceMeWrapper::SetMetadata)
      .def_static("is_enabled", &TraceMeWrapper::IsEnabled);

  m.def(
      "get_profiled_instructions_proto",
      [](std::string tensorboard_dir) -> nb::bytes {
        tensorflow::profiler::ProfiledInstructionsProto profile_proto;
        zkx::ThrowIfError(
            zkx::ConvertXplaneUnderLogdirToProfiledInstructionsProto(
                tensorboard_dir, &profile_proto));
        std::string profile_proto_str = profile_proto.SerializeAsString();
        return nb::bytes(profile_proto_str.data(), profile_proto_str.size());
      },
      nb::arg("tensorboard_dir"));

  m.def(
      "get_instructions_profile",
      [](const std::string& tensorboard_dir)
          -> std::vector<std::pair<std::string, double>> {
        tensorflow::profiler::ProfiledInstructionsProto profile_proto;
        zkx::ThrowIfError(
            zkx::ConvertXplaneUnderLogdirToProfiledInstructionsProto(
                tensorboard_dir, &profile_proto));
        std::vector<std::pair<std::string, double>> results;
        results.reserve(profile_proto.costs().size());
        for (const auto& c : profile_proto.costs()) {
          results.emplace_back(c.name(), c.cost_us());
        }
        return results;
      },
      nb::arg("tensorboard_dir"));

  m.def("get_fdo_profile",
        [](nb::bytes xspace, bool as_textproto = false) -> nb::object {
          std::string out = GetFdoProfile(
              std::string(xspace.c_str(), xspace.size()), as_textproto);
          return nb::bytes(out.data(), out.size());
        });

  m.def("get_fdo_profile", [](nb::bytes xspace) -> nb::object {
    std::string out = GetFdoProfile(std::string(xspace.c_str(), xspace.size()));
    return nb::bytes(out.data(), out.size());
  });

  m.def(
      "aggregate_profiled_instructions",
      [](const std::vector<nb::bytes>& profiles, int percentile) -> nb::object {
        std::vector<tensorflow::profiler::ProfiledInstructionsProto>
            fdo_profiles;
        for (const nb::bytes& profile : profiles) {
          tensorflow::profiler::ProfiledInstructionsProto profile_proto;
          profile_proto.ParseFromString(
              std::string(profile.c_str(), profile.size()));
          fdo_profiles.push_back(std::move(profile_proto));
        }

        tensorflow::profiler::ProfiledInstructionsProto result_proto;
        zkx::AggregateProfiledInstructionsProto(fdo_profiles, percentile,
                                                &result_proto);
        auto result = result_proto.SerializeAsString();
        return nb::bytes(result.data(), result.size());
      },
      nb::arg("profiles") = nb::list(), nb::arg("percentile"));
}

}  // namespace zkx
