// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_GESTURE_DET__GESTURE_DET_BACKEND_HPP_
#define TOPSBOT_GESTURE_DET__GESTURE_DET_BACKEND_HPP_

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <ta-runtime-api.h>

#include "topsbot_ai_infer/inference_backend.hpp"
#include "topsbot_ai_infer/object_types.hpp"

namespace topsbot_gesture_det
{

// ── Constants ─────────────────────────────────────────────────────────────
static constexpr int PALM_NUM_BOXES      = 2016;
static constexpr int PALM_NUM_COORDS     = 18;
static constexpr int PALM_NUM_KEYPOINTS  = 7;
static constexpr int PALM_MAX_RESULT     = 10;
static constexpr int HAND_NUM_LANDMARKS  = 21;
static constexpr int HAND_LANDMARK_ELEMS = 63;  // 21 * 3

// ── Anchor entry ──────────────────────────────────────────────────────────
struct PalmAnchor { float x; float y; };

/// Returns the precomputed 2016-entry anchor table.
const PalmAnchor * palm_anchors();

// ── Internal data types ───────────────────────────────────────────────────
struct PalmObject
{
  cv::Rect2f                              box;
  float                                   score{0.0f};
  std::array<cv::Point2f, PALM_NUM_KEYPOINTS> kpts{};
};

struct HandResult
{
  cv::Rect2f                                   box;
  float                                        score{0.0f};
  float                                        handedness{0.0f};
  std::array<cv::Point3f, HAND_NUM_LANDMARKS>  landmarks{};
  std::string                                  gesture;
};

// ── ModelContext – wraps one loaded TA-Runtime model ──────────────────────
struct ModelContext
{
  ta_runtime_context               context{0};
  int                              input_num{0};
  int                              output_num{0};
  std::vector<taconn_inout_attr_t> ins_attr;
  std::vector<taconn_inout_attr_t> outs_attr;
  taconn_input_t *                 input_tensors{nullptr};
  taconn_buffer_t *                output_buffer{nullptr};
  std::vector<std::vector<float>>  output_fp32_cache;
  int                              input_width{0};
  int                              input_height{0};
};

// ── Backend ───────────────────────────────────────────────────────────────
class GestureDetBackend final : public topsbot_ai_infer::IInferenceBackend
{
public:
  ~GestureDetBackend() override { shutdown(); }

  const char * name() const noexcept override { return "gesture_det"; }

  bool initialize(
    const topsbot_ai_infer::ModelConfig & cfg, std::string & error) override;

  topsbot_ai_infer::InferenceResult infer(
    const topsbot_ai_infer::InferenceFrame & frame) override;

private:
  void shutdown();

  bool load_model(ModelContext & ctx, const std::string & path, std::string & error);
  void release_model(ModelContext & ctx);
  bool run_model(ModelContext & ctx);

  // Stage-1 preprocessing: letterbox + resize + BGR→RGB using OpenCV.
  bool preprocess_palm(
    const cv::Mat & bgr,
    float & ratio, cv::Point2i & pad);

  void post_process_palm(
    const cv::Mat & image,
    float ratio, cv::Point2i pad,
    float conf_thresh, float nms_thresh,
    std::vector<PalmObject> & palms);

  bool preprocess_landmark(
    const cv::Mat & image, const PalmObject & palm,
    cv::Mat & out_input, cv::Rect2f & rotated_bbox,
    float & angle, cv::Matx23f & rot_mat, cv::Point2f & pad_bias);

  bool post_process_landmark(
    const PalmObject & palm,
    const cv::Rect2f & rotated_bbox, float angle,
    const cv::Matx23f & rot_mat, const cv::Point2f & pad_bias,
    float conf_thresh, HandResult & hand);

  void fill_input_tensor(ModelContext & ctx, const cv::Mat & rgb_input, int idx = 0);
  const float * output_to_fp32(ModelContext & ctx, int out_idx);

  // Convert InferenceFrame (any format) to a BGR cv::Mat.
  cv::Mat frame_to_bgr(const topsbot_ai_infer::InferenceFrame & frame);

  // Smooth gesture output with a sliding-window majority vote.
  // Key = per-hand tracking index (0-based order in the frame).
  // Value = ring buffer of the last GESTURE_SMOOTH_WINDOW raw gesture labels.
  static constexpr int GESTURE_SMOOTH_WINDOW = 5;
  std::string smooth_gesture(int hand_idx, const std::string & raw);
  std::unordered_map<int, std::deque<std::string>> gesture_history_;

  topsbot_ai_infer::ModelConfig config_;
  ModelContext palm_ctx_;
  ModelContext landmark_ctx_;
  bool         runtime_acquired_{false};
};

}  // namespace topsbot_gesture_det

#endif  // TOPSBOT_GESTURE_DET__GESTURE_DET_BACKEND_HPP_
