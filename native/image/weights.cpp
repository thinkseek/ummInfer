// SPDX-License-Identifier: Apache-2.0
#include "weights.hpp"

namespace umminfer::image {

DeviceTensor load_bf16(const ShardedSafeTensors& st, std::string_view name) {
  const StTensor& info = st.info(name);
  if (info.dtype != "BF16")
    throw std::invalid_argument("expected BF16 tensor: " + std::string(name));
  DeviceTensor t(std::vector<int64_t>(info.shape.begin(), info.shape.end()));
  t.copy_from_host(st.host_ptr(info));
  return t;
}

}  // namespace umminfer::image
