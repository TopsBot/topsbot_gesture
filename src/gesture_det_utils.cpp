// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_gesture_det/gesture_det_utils.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace topsbot_gesture_det
{

size_t tensor_element_count(const taconn_inout_attr_t & attr)
{
  size_t count = 1;
  for (unsigned int i = 0; i < attr.dim_count; ++i) {
    if (attr.dim_size[i] == 0) { return 0; }
    count *= static_cast<size_t>(attr.dim_size[i]);
  }
  return count;
}

namespace
{
size_t element_size(unsigned int format)
{
  switch (format) {
    case TACONN_DATA_FORMAT_FP32: return sizeof(float);
    case TACONN_DATA_FORMAT_FP16: return sizeof(uint16_t);
    case TACONN_DATA_FORMAT_UINT8:
    case TACONN_DATA_FORMAT_INT8: return sizeof(uint8_t);
    default: return 0;
  }
}
}  // namespace

size_t tensor_byte_size(const taconn_inout_attr_t & attr)
{
  return tensor_element_count(attr) * element_size(attr.data_format);
}

uint16_t float_to_half(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFU;
  if (exp <= 0) {
    if (exp < -10) { return static_cast<uint16_t>(sign); }
    mant = (mant | 0x800000U) >> static_cast<uint32_t>(1 - exp);
    if ((mant & 0x1000U) != 0U) { mant += 0x2000U; }
    return static_cast<uint16_t>(sign | (mant >> 13U));
  }
  if (exp >= 31) { return static_cast<uint16_t>(sign | 0x7C00U); }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | (mant >> 13U));
}

float half_to_float(uint16_t value)
{
  const uint32_t sign = static_cast<uint32_t>(value >> 15U) << 31U;
  uint32_t exp = (value >> 10U) & 0x1FU;
  uint32_t mant = value & 0x3FFU;
  if (exp == 0) {
    if (mant == 0) {
      uint32_t bits = sign; float f; std::memcpy(&f, &bits, 4); return f;
    }
    exp = 1;
    while ((mant & 0x400U) == 0U) { mant <<= 1U; --exp; }
    mant &= 0x3FFU;
    exp += 127 - 15;
  } else if (exp == 31) {
    exp = 0xFF;
  } else {
    exp += 127 - 15;
  }
  uint32_t bits = sign | (exp << 23U) | (mant << 13U);
  float f; std::memcpy(&f, &bits, 4); return f;
}

float decode_tensor_value(const void * data, size_t index, const taconn_inout_attr_t & attr)
{
  switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP32:
      return static_cast<const float *>(data)[index];
    case TACONN_DATA_FORMAT_FP16:
      return half_to_float(static_cast<const uint16_t *>(data)[index]);
    case TACONN_DATA_FORMAT_UINT8: {
      const float raw = static_cast<float>(static_cast<const uint8_t *>(data)[index]);
      if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
        return (raw - static_cast<float>(attr.quant_data.affine.tf_zero_point)) *
               attr.quant_data.affine.tf_scale;
      }
      if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
        return std::ldexp(raw, -attr.quant_data.dfp.fixed_point_pos);
      }
      return raw;
    }
    case TACONN_DATA_FORMAT_INT8: {
      const float raw = static_cast<float>(static_cast<const int8_t *>(data)[index]);
      if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
        return (raw - static_cast<float>(attr.quant_data.affine.tf_zero_point)) *
               attr.quant_data.affine.tf_scale;
      }
      if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
        return std::ldexp(raw, -attr.quant_data.dfp.fixed_point_pos);
      }
      return raw;
    }
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

std::vector<float> tensor_to_float32(
  const void * data, size_t num_elements, const taconn_inout_attr_t & attr)
{
  std::vector<float> out(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = decode_tensor_value(data, i, attr);
  }
  return out;
}

namespace
{
uint64_t access_flags(const CpuAccess access)
{
  switch (access) {
    case CpuAccess::Read:
      return DMA_BUF_SYNC_READ;
    case CpuAccess::Write:
      return DMA_BUF_SYNC_WRITE;
  }
  return DMA_BUF_SYNC_RW;
}
}  // namespace

// Implementation must be at topsbot_gesture_det:: scope (matching header decl),
// not inside an anonymous namespace, otherwise it would be hidden.
bool dmabuf_cpu_sync(int fd, CpuAccess access, bool start)
{
  if (fd < 0) {
    return false;
  }
  dma_buf_sync sync{};
  sync.flags = access_flags(access) | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END);
  int result = -1;
  do {
    result = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
  } while (result < 0 && errno == EINTR);
  return result == 0;
}

}  // namespace topsbot_gesture_det
