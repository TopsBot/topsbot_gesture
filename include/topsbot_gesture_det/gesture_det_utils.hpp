// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_GESTURE_DET__GESTURE_DET_UTILS_HPP_
#define TOPSBOT_GESTURE_DET__GESTURE_DET_UTILS_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ta-runtime-api.h>

#include "topsbot_ai_infer/inference_backend.hpp"

namespace topsbot_gesture_det
{

/// Returns total element count of a tensor (product of all dims).
size_t tensor_element_count(const taconn_inout_attr_t & attr);

/// Returns byte size of a tensor buffer.
size_t tensor_byte_size(const taconn_inout_attr_t & attr);

/// Convert float32 to float16 bit pattern.
uint16_t float_to_half(float value);

/// Convert float16 bit pattern to float32.
float half_to_float(uint16_t value);

/// Decode a single element from a tensor buffer to float32 (handles FP32/FP16/INT8/UINT8 + quant).
float decode_tensor_value(
  const void * data, size_t index, const taconn_inout_attr_t & attr);

/// Convert a full output tensor to a float32 vector (handles quantization).
std::vector<float> tensor_to_float32(
  const void * data, size_t num_elements, const taconn_inout_attr_t & attr);

/// Sigmoid activation.
inline float sigmoid(float x)
{
  if (x < -100.0F) { return 0.0F; }
  if (x > 100.0F) { return 1.0F; }
  return 1.0F / (1.0F + std::exp(-x));
}

enum class CpuAccess
{
  Read = 0,
  Write = 1
};

bool dmabuf_cpu_sync(int fd, CpuAccess access, bool start);

}  // namespace topsbot_gesture_det

#endif  // TOPSBOT_GESTURE_DET__GESTURE_DET_UTILS_HPP_
