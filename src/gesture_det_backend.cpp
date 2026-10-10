// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_gesture_det/gesture_det_backend.hpp"
#include "topsbot_gesture_det/gesture_det_utils.hpp"
#include "topsbot_gesture_det/anchors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <ta-runtime-api.h>
#include <pluginlib/class_list_macros.hpp>

#include "topsbot_ai_infer/inference_backend.hpp"
#include "topsbot_ai_infer/object_types.hpp"

namespace topsbot_gesture_det
{

using topsbot_ai_infer::IInferenceBackend;
using topsbot_ai_infer::InferenceFrame;
using topsbot_ai_infer::InferenceResult;
using topsbot_ai_infer::ModelConfig;
using topsbot_ai_infer::PixelFormat;
using topsbot_ai_infer::LandmarkMeta;
using topsbot_ai_infer::ClassifierMeta;
using topsbot_ai_infer::MetaType;
using topsbot_ai_infer::ObjectMeta;
using topsbot_ai_infer::FrameMeta;

namespace
{

using SteadyClock = std::chrono::steady_clock;

float elapsed_ms(SteadyClock::time_point a, SteadyClock::time_point b)
{
  return std::chrono::duration<float, std::milli>(b - a).count();
}

std::string expand_user_path(const std::string & path)
{
  if (path.empty() || path.front() != '~') {
    return path;
  }
  const char * user_home = std::getenv("HOME");
  if (user_home == nullptr || user_home[0] == '\0') {
    return path;
  }
  if (path.size() == 1U || path[1] == '/') {
    return std::string(user_home) + path.substr(1);
  }
  return path;
}

// ── Runtime reference-count (shared across all backend instances) ──────────
std::mutex runtime_mutex;
size_t runtime_users = 0;

bool acquire_runtime()
{
  std::lock_guard<std::mutex> lk(runtime_mutex);
  if (runtime_users == 0 && ta_runtime_init() != NNRT_SUCCESS) {
    return false;
  }
  ++runtime_users;
  return true;
}

void release_runtime()
{
  std::lock_guard<std::mutex> lk(runtime_mutex);
  if (runtime_users == 0) { return; }
  if (--runtime_users == 0) { (void)ta_runtime_deinit(); }
}

// ── Geometry helpers ──────────────────────────────────────────────────────
cv::Point2f transform_point(const cv::Matx23f & m, const cv::Point2f & p)
{
  return {m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2),
          m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2)};
}

cv::Matx23f rotation_matrix_2d(const cv::Point2f & center, float angle_deg, float scale)
{
  cv::Mat m = cv::getRotationMatrix2D(center, angle_deg, scale);
  return cv::Matx23f(
    static_cast<float>(m.at<double>(0, 0)), static_cast<float>(m.at<double>(0, 1)),
    static_cast<float>(m.at<double>(0, 2)),
    static_cast<float>(m.at<double>(1, 0)), static_cast<float>(m.at<double>(1, 1)),
    static_cast<float>(m.at<double>(1, 2)));
}

cv::Rect2f bbox_from_points(const std::vector<cv::Point2f> & points)
{
  if (points.empty()) { return {}; }
  float min_x = points[0].x;
  float min_y = points[0].y;
  float max_x = points[0].x;
  float max_y = points[0].y;
  for (const auto & point : points) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  return {min_x, min_y, max_x - min_x, max_y - min_y};
}

constexpr int PALM_BASE_KEYPOINT = 0;
constexpr int MIDDLE_FINGER_BASE_KEYPOINT = 2;
const cv::Point2f PALM_BOX_PRE_SHIFT(0.0F, 0.0F);
constexpr float PALM_BOX_PRE_ENLARGE = 4.0F;
const cv::Point2f PALM_BOX_SHIFT(0.0F, -0.4F);
constexpr float PALM_BOX_ENLARGE = 3.0F;
const cv::Point2f HAND_BOX_SHIFT(0.0F, -0.1F);
constexpr float HAND_BOX_ENLARGE = 1.65F;

bool crop_and_pad_from_palm(
  const cv::Mat & image, const cv::Rect2f & input_bbox, bool for_rotation,
  cv::Mat & output, cv::Rect2f & output_bbox, cv::Point2f & bias)
{
  if (image.empty()) { return false; }

  cv::Point2f top_left(input_bbox.x, input_bbox.y);
  cv::Point2f bottom_right(
    input_bbox.x + input_bbox.width, input_bbox.y + input_bbox.height);
  cv::Point2f size = bottom_right - top_left;
  const cv::Point2f shift = for_rotation ? PALM_BOX_PRE_SHIFT : PALM_BOX_SHIFT;
  top_left.x += shift.x * size.x;
  top_left.y += shift.y * size.y;
  bottom_right.x += shift.x * size.x;
  bottom_right.y += shift.y * size.y;

  const cv::Point2f center = (top_left + bottom_right) * 0.5F;
  size = bottom_right - top_left;
  const float enlarge = for_rotation ? PALM_BOX_PRE_ENLARGE : PALM_BOX_ENLARGE;
  const cv::Point2f half = size * (enlarge * 0.5F);
  top_left = center - half;
  bottom_right = center + half;

  int x1 = std::max(0, std::min(static_cast<int>(top_left.x), image.cols));
  int y1 = std::max(0, std::min(static_cast<int>(top_left.y), image.rows));
  int x2 = std::max(0, std::min(static_cast<int>(bottom_right.x), image.cols));
  int y2 = std::max(0, std::min(static_cast<int>(bottom_right.y), image.rows));
  if (x2 <= x1 || y2 <= y1) { return false; }

  cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
  int side = for_rotation ?
    static_cast<int>(std::sqrt(static_cast<float>(
      crop.rows * crop.rows + crop.cols * crop.cols))) :
    std::max(crop.rows, crop.cols);
  side = std::max(side, 1);

  const int pad_h = side - crop.rows;
  const int pad_w = side - crop.cols;
  const int left = pad_w / 2;
  const int top = pad_h / 2;
  cv::copyMakeBorder(
    crop, output, top, pad_h - top, left, pad_w - left,
    cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  output_bbox = cv::Rect2f(
    static_cast<float>(x1), static_cast<float>(y1),
    static_cast<float>(x2 - x1), static_cast<float>(y2 - y1));
  bias = cv::Point2f(static_cast<float>(x1 - left), static_cast<float>(y1 - top));
  return true;
}

cv::Rect2f clamp_rect_to_image(const cv::Rect2f & rect, const cv::Size & image_size)
{
  const float x1 = std::max(0.0F, std::min(rect.x, static_cast<float>(image_size.width)));
  const float y1 = std::max(0.0F, std::min(rect.y, static_cast<float>(image_size.height)));
  const float x2 = std::max(
    x1, std::min(rect.x + rect.width, static_cast<float>(image_size.width)));
  const float y2 = std::max(
    y1, std::min(rect.y + rect.height, static_cast<float>(image_size.height)));
  return {x1, y1, x2 - x1, y2 - y1};
}

// ── NMS ───────────────────────────────────────────────────────────────────
float rect_iou(const cv::Rect2f & a, const cv::Rect2f & b)
{
  const float ix1 = std::max(a.x, b.x);
  const float iy1 = std::max(a.y, b.y);
  const float ix2 = std::min(a.x + a.width,  b.x + b.width);
  const float iy2 = std::min(a.y + a.height, b.y + b.height);
  const float iw  = std::max(0.0f, ix2 - ix1);
  const float ih  = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
  const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
  const float uni = area_a + area_b - inter;
  return (uni <= 0.0f) ? 0.0f : inter / uni;
}

void palm_nms(
  const std::vector<cv::Rect2f> & boxes,
  const std::vector<float> & scores,
  std::vector<int> & picked,
  float nms_threshold,
  int top_k = PALM_MAX_RESULT)
{
  picked.clear();
  const int n = static_cast<int>(boxes.size());
  if (n <= 0) { return; }

  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
    [&](int a, int b) { return scores[a] > scores[b]; });

  for (int idx : order) {
    bool keep = true;
    for (int kept : picked) {
      if (rect_iou(boxes[idx], boxes[kept]) > nms_threshold) { keep = false; break; }
    }
    if (keep) {
      picked.push_back(idx);
      if (top_k > 0 && static_cast<int>(picked.size()) >= top_k) { break; }
    }
  }
}

// ── Gesture recognition (rule-based) ─────────────────────────────────────
//
// Four fingers (index/middle/ring/pinky) use joint-angle cosine:
//   extended = both PIP and DIP joint angles are near-straight (cos > threshold)
//   This is scale-invariant (works at any distance from camera).
//
// Thumb uses a distance-ratio method instead of joint-angle, because the
// thumb's bending axis is perpendicular to the palm plane — its 2-D projected
// joint angles are unreliable and cause HANG_LOOSE / POINT confusion:
//   extended = dist(tip=4, index_mcp=5) > dist(ip=3, index_mcp=5) * THUMB_RATIO
//   This works regardless of hand orientation and camera distance.
//
// All coordinates are 2-D (x,y) only; landmark z is not reliable after
// affine back-projection.
std::string recognize_gesture(
  const std::array<cv::Point3f, HAND_NUM_LANDMARKS> & lm)
{
  // 2-D Euclidean distance helper
  auto dist2d = [&](int a, int b) -> float {
    const float dx = lm[static_cast<size_t>(a)].x - lm[static_cast<size_t>(b)].x;
    const float dy = lm[static_cast<size_t>(a)].y - lm[static_cast<size_t>(b)].y;
    return std::sqrt(dx * dx + dy * dy);
  };

  // Joint-angle cosine at vertex b along chain a→b→c.
  // Near 1.0 = nearly straight segment; small/negative = bent.
  auto joint_cos = [&](int a, int b, int c) -> float {
    const float v1x = lm[static_cast<size_t>(b)].x - lm[static_cast<size_t>(a)].x;
    const float v1y = lm[static_cast<size_t>(b)].y - lm[static_cast<size_t>(a)].y;
    const float v2x = lm[static_cast<size_t>(c)].x - lm[static_cast<size_t>(b)].x;
    const float v2y = lm[static_cast<size_t>(c)].y - lm[static_cast<size_t>(b)].y;
    const float dot = v1x * v2x + v1y * v2y;
    const float len = std::sqrt((v1x * v1x + v1y * v1y) * (v2x * v2x + v2y * v2y));
    return (len > 1e-6f) ? (dot / len) : 1.0f;
  };

  // Landmark indices per finger:
  //   index:  MCP=5  PIP=6  DIP=7  TIP=8
  //   middle: MCP=9  PIP=10 DIP=11 TIP=12
  //   ring:   MCP=13 PIP=14 DIP=15 TIP=16
  //   pinky:  MCP=17 PIP=18 DIP=19 TIP=20
  //   thumb:  CMC=1  MCP=2  IP=3   TIP=4
  constexpr float STRAIGHT_COS = 0.7f;   // cos(45°): both PIP+DIP must exceed this

  bool index  = joint_cos(5,  6,  7)  > STRAIGHT_COS &&
                joint_cos(6,  7,  8)  > STRAIGHT_COS;
  bool middle = joint_cos(9,  10, 11) > STRAIGHT_COS &&
                joint_cos(10, 11, 12) > STRAIGHT_COS;
  bool ring   = joint_cos(13, 14, 15) > STRAIGHT_COS &&
                joint_cos(14, 15, 16) > STRAIGHT_COS;
  bool pinky  = joint_cos(17, 18, 19) > STRAIGHT_COS &&
                joint_cos(18, 19, 20) > STRAIGHT_COS;

  // Thumb: distance-ratio method — scale & orientation invariant.
  // tip(4) farther from index_mcp(5) than ip(3) is → thumb is open/extended.
  constexpr float THUMB_RATIO = 1.15f;
  bool thumb = dist2d(4, 5) > dist2d(3, 5) * THUMB_RATIO;

  int count = (thumb ? 1 : 0) + (index ? 1 : 0) + (middle ? 1 : 0) +
              (ring ? 1 : 0)  + (pinky ? 1 : 0);

  if (count == 0)  { return "FIST"; }
  if (count == 5)  { return "OPEN"; }
  // POINT: only index extended (thumb state don't care — avoids thumb confusion)
  if (index && !middle && !ring && !pinky) { return "POINT"; }
  if (index && middle && !ring && !pinky)  { return "VICTORY"; }
  // HANG_LOOSE: thumb AND pinky extended, other three curled
  if (thumb && pinky && !index && !middle && !ring) { return "HANG_LOOSE"; }
  return "UNKNOWN";
}

}  // anonymous namespace

// ──────────────────────────────────────────────────────────────────────────
// GestureDetBackend implementation
// ──────────────────────────────────────────────────────────────────────────

// ── smooth_gesture ──────────────────────────────────────────────────────────
// Two-stage filter:
//   1. Streak gate  – only switch to a new label when it appears in
//      GESTURE_CONFIRM_STREAK consecutive frames.  Single-frame noise
//      (background face, bad angle) is suppressed without adding latency
//      to genuine transitions.
//   2. Majority fallback – if the streak gate hasn't fired yet, return the
//      majority-vote winner of the sliding window so the output is never
//      "stuck" on stale data during a real transition.
std::string GestureDetBackend::smooth_gesture(
  int hand_idx, const std::string & raw)
{
  auto & hist = gesture_history_[hand_idx];
  hist.push_back(raw);
  if (static_cast<int>(hist.size()) > GESTURE_SMOOTH_WINDOW) {
    hist.pop_front();
  }

  // Stage 1: streak gate — count how many trailing frames match the latest.
  constexpr int GESTURE_CONFIRM_STREAK = 3;
  int streak = 0;
  for (auto it = hist.rbegin(); it != hist.rend() && *it == hist.back(); ++it) {
    ++streak;
  }
  if (streak >= GESTURE_CONFIRM_STREAK) {
    return hist.back();
  }

  // Stage 2: majority vote (streak not yet met — keep last stable output).
  // Iterate deque in chronological order so later occurrences win ties.
  std::unordered_map<std::string, int> counts;
  for (const auto & g : hist) { counts[g]++; }
  std::string best = raw;
  int best_count = 0;
  for (const auto & g : hist) {
    int c = counts[g];
    if (c >= best_count) {
      best_count = c;
      best = g;
    }
  }
  return best;
}

// ── initialize ──────────────────────────────────────────────────────────────
bool GestureDetBackend::initialize(const ModelConfig & cfg, std::string & error)
{
  shutdown();
  error.clear();
  config_ = cfg;

  if (config_.model_path.empty() ||
    !std::filesystem::is_regular_file(config_.model_path))
  {
    error = "gesture_det: palm model file not found: " + config_.model_path;
    return false;
  }

  if (!acquire_runtime()) {
    error = "gesture_det: ta_runtime_init failed";
    return false;
  }
  runtime_acquired_ = true;

  if (!load_model(palm_ctx_, config_.model_path, error)) { shutdown(); return false; }
  bool has_palm_boxes = false;
  bool has_palm_scores = false;
  for (const auto & attr : palm_ctx_.outs_attr) {
    const size_t elements = tensor_element_count(attr);
    has_palm_boxes = has_palm_boxes ||
      elements == static_cast<size_t>(PALM_NUM_BOXES * PALM_NUM_COORDS);
    has_palm_scores = has_palm_scores || elements == static_cast<size_t>(PALM_NUM_BOXES);
  }
  if (palm_ctx_.input_num != 1 || palm_ctx_.input_width != 192 ||
    palm_ctx_.input_height != 192 || !has_palm_boxes || !has_palm_scores)
  {
    error = "gesture_det: palm model contract must be 192x192 RGB with "
      "2016x18 boxes and 2016 scores";
    shutdown(); return false;
  }

  if (config_.class_names.empty() || config_.class_names[0].empty()) {
    error = "gesture_det: landmark model path is required in postprocess.class_names[0]";
    shutdown(); return false;
  }
  const std::string lm_path = expand_user_path(config_.class_names[0]);
  if (!std::filesystem::is_regular_file(lm_path)) {
    error = "gesture_det: landmark model file not found: " + lm_path;
    shutdown(); return false;
  }
  if (!load_model(landmark_ctx_, lm_path, error)) { shutdown(); return false; }

  if (landmark_ctx_.input_num != 1 || landmark_ctx_.input_width != 224 ||
    landmark_ctx_.input_height != 224 || landmark_ctx_.output_num != 3 ||
    tensor_element_count(landmark_ctx_.outs_attr[0]) != HAND_LANDMARK_ELEMS ||
    tensor_element_count(landmark_ctx_.outs_attr[1]) != 1U ||
    tensor_element_count(landmark_ctx_.outs_attr[2]) != 1U)
  {
    error = "gesture_det: landmark model contract must be 224x224 RGB with "
      "outputs ordered as 63, 1, 1 elements";
    shutdown(); return false;
  }

  return true;
}

// ── infer ────────────────────────────────────────────────────────────────────
InferenceResult GestureDetBackend::infer(const InferenceFrame & frame)
{
  InferenceResult result;
  result.backend    = name();
  result.model_name = config_.model_name;
  result.device_id  = config_.device_id;

  auto t0 = SteadyClock::now();

  // Convert InferenceFrame to cv::Mat (BGR) — handles BGR8, NV12, JPEG
  cv::Mat bgr = frame_to_bgr(frame);
  if (bgr.empty()) {
    result.success = false;
    result.error   = "gesture_det: failed to decode input frame";
    return result;
  }

  // ── Stage 1: palm detection ─────────────────────────────────────────────
  float ratio = 1.0f;
  cv::Point2i pad(0, 0);
  if (!preprocess_palm(bgr, ratio, pad)) {
    result.success = false;
    result.error   = "gesture_det: palm preprocessing failed";
    return result;
  }

  if (!run_model(palm_ctx_)) {
    result.success = false;
    result.error   = "gesture_det: palm model inference failed";
    return result;
  }

  std::vector<PalmObject> palms;
  post_process_palm(bgr, ratio, pad,
    config_.confidence_threshold, config_.nms_threshold, palms);

  auto t1 = SteadyClock::now();
  result.timing.preprocess_ms = elapsed_ms(t0, t1);
  result.timing.input_copy_ms = 0.0F;
  result.timing.accelerator_preprocess_ms = 0.0F;

  // ── Stage 2: per-hand landmark (OpenCV warpAffine) ──────────────────────
  // Trim history slots that exceed the current detected hand count.
  // Clearing only excess entries (rather than the whole map) avoids flushing
  // the smoothing window every time the detector briefly misses a hand for
  // one frame (which was the cause of "result flickers when hand is still").
  // Remove history slots whose index is >= the number of hands detected this
  // frame.  We only erase excess slots — NOT the whole map — so that a brief
  // one-frame miss (palm score drops just below threshold) does not reset the
  // smoothing window and cause visible result flicker.
  const int n_palms = static_cast<int>(palms.size());
  for (auto it = gesture_history_.begin(); it != gesture_history_.end(); ) {
    if (it->first >= n_palms) {
      it = gesture_history_.erase(it);
    } else {
      ++it;
    }
  }
  int hand_idx = 0;
  for (const auto & palm : palms) {
    ObjectMeta obj;
    obj.type        = topsbot_ai_infer::ObjectType::Unknown;
    obj.class_name  = "hand";
    obj.confidence  = palm.score;
    const cv::Rect2f visible_palm = clamp_rect_to_image(palm.box, bgr.size());
    obj.detector_bbox.x      = visible_palm.x;
    obj.detector_bbox.y      = visible_palm.y;
    obj.detector_bbox.width  = visible_palm.width;
    obj.detector_bbox.height = visible_palm.height;
    obj.display_rect = obj.detector_bbox;

    if (landmark_ctx_.context != 0) {
      cv::Mat lm_rgb;
      cv::Rect2f rot_bbox;
      float angle = 0.0f;
      cv::Matx23f rot_mat;
      cv::Point2f pad_bias;

      if (preprocess_landmark(bgr, palm, lm_rgb, rot_bbox, angle, rot_mat, pad_bias)) {
        fill_input_tensor(landmark_ctx_, lm_rgb);

        if (run_model(landmark_ctx_)) {
          HandResult hand;
          if (post_process_landmark(palm, rot_bbox, angle, rot_mat, pad_bias,
                config_.confidence_threshold, hand))
          {
            const cv::Rect2f visible_hand = clamp_rect_to_image(hand.box, bgr.size());
            obj.detector_bbox.x = visible_hand.x;
            obj.detector_bbox.y = visible_hand.y;
            obj.detector_bbox.width = visible_hand.width;
            obj.detector_bbox.height = visible_hand.height;
            obj.display_rect = obj.detector_bbox;
            obj.confidence = hand.score;

            // Gesture classifier meta
            ClassifierMeta gesture_meta;
            gesture_meta.attr_id   = 0;
            // Label kept raw here; ai_infer rewrites it as "h{idx}:{label}"
            // using its ObjectMeta index, so the frontend only needs to know
            // how to split on ':'.
            gesture_meta.label     = smooth_gesture(hand_idx, hand.gesture);
            gesture_meta.confidence = hand.score;
            obj.attributes.push_back(gesture_meta);

            // Handedness classifier meta
            ClassifierMeta hand_meta;
            hand_meta.attr_id   = 1;
            hand_meta.label     = (hand.handedness > 0.5f) ? "Right" : "Left";
            hand_meta.confidence = hand.handedness;
            obj.attributes.push_back(hand_meta);

            // Landmark user meta
            auto lm_meta = std::make_shared<LandmarkMeta>();
            lm_meta->model_name = "hand_kps";
            lm_meta->confidence = hand.score;
            lm_meta->points.resize(HAND_NUM_LANDMARKS);
            for (int i = 0; i < HAND_NUM_LANDMARKS; ++i) {
              const auto & lp = hand.landmarks[static_cast<size_t>(i)];
              lm_meta->points[static_cast<size_t>(i)].x = lp.x;
              lm_meta->points[static_cast<size_t>(i)].y = lp.y;
              lm_meta->points[static_cast<size_t>(i)].confidence = hand.score;
            }
            obj.user_metas.push_back(lm_meta);
          }
        }
      }
    }

    ++hand_idx;
    result.frame.objects.push_back(std::move(obj));
  }

  // If this frame had no palm detections, still publish a placeholder
  // object carrying an attribute with attr_id=0 so that topsbot_ai_infer
  // forwards a (label="none") /classifications_by_object message for every
  // frame. topsbot_webviz's process_loop drops frames whose stamp cannot
  // be matched with a classification, which on a board without reliable
  // libswscale and rare hand appearances would otherwise leave the
  // browser image stream frozen. The placeholder is harmless to other
  // downstream consumers: it has no detector_bbox / user_metas and a
  // confidence of 0.0, so they can filter it out by score threshold or
  // by the special "none" label.
  if (result.frame.objects.empty()) {
    ObjectMeta placeholder;
    placeholder.type        = topsbot_ai_infer::ObjectType::Unknown;
    placeholder.confidence  = 0.0F;
    ClassifierMeta gesture_meta;
    gesture_meta.attr_id   = 0;
    gesture_meta.label     = "none";
    gesture_meta.confidence = 0.0F;
    placeholder.attributes.push_back(gesture_meta);
    result.frame.objects.push_back(std::move(placeholder));
  }

  auto t2 = SteadyClock::now();
  result.timing.postprocess_ms = elapsed_ms(t1, t2);
  result.timing.infer_ms       = elapsed_ms(t0, t2);
  result.success = true;
  return result;
}

// ── shutdown ─────────────────────────────────────────────────────────────────
void GestureDetBackend::shutdown()
{
  release_model(palm_ctx_);
  release_model(landmark_ctx_);
  if (runtime_acquired_) {
    release_runtime();
    runtime_acquired_ = false;
  }
}

// ── load_model ────────────────────────────────────────────────────────────────
bool GestureDetBackend::load_model(
  ModelContext & ctx, const std::string & path, std::string & error)
{
  if (ta_runtime_load_model_from_file(
        &ctx.context, path.c_str(),
        static_cast<uint32_t>(config_.device_id)) != NNRT_SUCCESS)
  {
    error = "gesture_det: ta_runtime_load_model_from_file failed for " + path;
    return false;
  }

  taconn_input_output_num_t counts{};
  if (ta_runtime_query(&ctx.context, TACONN_QUERY_IN_OUT_NUM, &counts) != NNRT_SUCCESS) {
    error = "gesture_det: ta_runtime_query IN_OUT_NUM failed";
    return false;
  }
  ctx.input_num  = static_cast<int>(counts.input_num);
  ctx.output_num = static_cast<int>(counts.output_num);

  ctx.ins_attr.resize(static_cast<size_t>(ctx.input_num));
  for (int i = 0; i < ctx.input_num; ++i) {
    ctx.ins_attr[static_cast<size_t>(i)] = {};
    ctx.ins_attr[static_cast<size_t>(i)].index = static_cast<uint32_t>(i);
    if (ta_runtime_query(&ctx.context, TACONN_QUERY_INPUT_ATTR,
          &ctx.ins_attr[static_cast<size_t>(i)]) != NNRT_SUCCESS)
    {
      error = "gesture_det: TACONN_QUERY_INPUT_ATTR failed";
      return false;
    }
  }

  ctx.outs_attr.resize(static_cast<size_t>(ctx.output_num));
  for (int i = 0; i < ctx.output_num; ++i) {
    ctx.outs_attr[static_cast<size_t>(i)] = {};
    ctx.outs_attr[static_cast<size_t>(i)].index = static_cast<uint32_t>(i);
    if (ta_runtime_query(&ctx.context, TACONN_QUERY_OUTPUT_ATTR,
          &ctx.outs_attr[static_cast<size_t>(i)]) != NNRT_SUCCESS)
    {
      error = "gesture_det: TACONN_QUERY_OUTPUT_ATTR failed";
      return false;
    }
  }

  // Infer spatial dims from first input tensor
  if (ctx.input_num > 0) {
    const auto & a = ctx.ins_attr[0];
    if (a.dim_count == 4) {
      // NHWC: dim_size = {N, H, W, C}; NCHW: {N, C, H, W}
      const bool is_nchw = (a.dim_size[1] == 3);
      ctx.input_height = static_cast<int>(is_nchw ? a.dim_size[2] : a.dim_size[1]);
      ctx.input_width  = static_cast<int>(is_nchw ? a.dim_size[3] : a.dim_size[2]);
    }
  }

  // Allocate and register input buffers
  ctx.input_tensors = static_cast<taconn_input_t *>(
    calloc(static_cast<size_t>(ctx.input_num), sizeof(taconn_input_t)));
  if (!ctx.input_tensors) { error = "gesture_det: alloc input_tensors failed"; return false; }

  for (int i = 0; i < ctx.input_num; ++i) {
    const size_t sz = tensor_byte_size(ctx.ins_attr[static_cast<size_t>(i)]);
    ctx.input_tensors[i].index = static_cast<uint32_t>(i);
    ctx.input_tensors[i].size  = static_cast<uint32_t>(sz);
    if (posix_memalign(reinterpret_cast<void **>(&ctx.input_tensors[i].data), 256, sz) != 0) {
      error = "gesture_det: posix_memalign failed"; return false;
    }
    memset(ctx.input_tensors[i].data, 0, sz);
  }
  if (ta_runtime_set_input_cva(&ctx.context, static_cast<uint32_t>(ctx.input_num),
        ctx.input_tensors) != NNRT_SUCCESS)
  {
    error = "gesture_det: ta_runtime_set_input_cva failed"; return false;
  }

  // Allocate and register output buffers
  ctx.output_buffer = static_cast<taconn_buffer_t *>(
    calloc(static_cast<size_t>(ctx.output_num), sizeof(taconn_buffer_t)));
  if (!ctx.output_buffer) { error = "gesture_det: alloc output_buffer failed"; return false; }

  for (int i = 0; i < ctx.output_num; ++i) {
    const size_t sz = tensor_byte_size(ctx.outs_attr[static_cast<size_t>(i)]);
    if (ta_runtime_create_buffer(&ctx.context, sz,
          &ctx.output_buffer[i]) != NNRT_SUCCESS)
    {
      error = "gesture_det: ta_runtime_create_buffer failed"; return false;
    }
  }
  if (ta_runtime_set_output(&ctx.context, static_cast<uint32_t>(ctx.output_num),
        ctx.output_buffer) != NNRT_SUCCESS)
  {
    error = "gesture_det: ta_runtime_set_output failed"; return false;
  }

  ctx.output_fp32_cache.resize(static_cast<size_t>(ctx.output_num));
  return true;
}

// ── release_model ─────────────────────────────────────────────────────────────
void GestureDetBackend::release_model(ModelContext & ctx)
{
  if (ctx.output_buffer) {
    for (int i = 0; i < ctx.output_num; ++i) {
      ta_runtime_destroy_buffer(&ctx.context, &ctx.output_buffer[i]);
    }
    free(ctx.output_buffer); ctx.output_buffer = nullptr;
  }
  if (ctx.input_tensors) {
    for (int i = 0; i < ctx.input_num; ++i) { free(ctx.input_tensors[i].data); }
    free(ctx.input_tensors); ctx.input_tensors = nullptr;
  }
  if (ctx.context) {
    ta_runtime_destroy_context(&ctx.context); ctx.context = 0;
  }
  ctx.ins_attr.clear(); ctx.outs_attr.clear(); ctx.output_fp32_cache.clear();
  ctx.input_num = 0; ctx.output_num = 0;
}

// ── run_model ────────────────────────────────────────────────────────────────
bool GestureDetBackend::run_model(ModelContext & ctx)
{
  if (ta_runtime_run_network(&ctx.context) != NNRT_SUCCESS) { return false; }
  for (int i = 0; i < ctx.output_num; ++i) {
    if (ta_runtime_invalidate_buffer(&ctx.context, &ctx.output_buffer[i]) != NNRT_SUCCESS) {
      return false;
    }
  }
  return true;
}

// ── preprocess_palm: OpenCV letterbox resize + BGR→RGB → fill tensor ─────────
bool GestureDetBackend::preprocess_palm(
  const cv::Mat & bgr,
  float & ratio, cv::Point2i & pad)
{
  const int tw = palm_ctx_.input_width  > 0 ? palm_ctx_.input_width  : config_.input_width;
  const int th = palm_ctx_.input_height > 0 ? palm_ctx_.input_height : config_.input_height;

  if (bgr.empty() || tw <= 0 || th <= 0) { return false; }

  const int src_w = bgr.cols;
  const int src_h = bgr.rows;
  const float r = std::min(
    static_cast<float>(tw) / static_cast<float>(src_w),
    static_cast<float>(th) / static_cast<float>(src_h));
  ratio = r;
  const int new_w = static_cast<int>(static_cast<float>(src_w) * r);
  const int new_h = static_cast<int>(static_cast<float>(src_h) * r);
  pad.x = (tw - new_w) / 2;
  pad.y = (th - new_h) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  cv::Mat letterboxed(th, tw, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(letterboxed(cv::Rect(pad.x, pad.y, new_w, new_h)));

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  fill_input_tensor(palm_ctx_, rgb, 0);
  return true;
}

// ── post_process_palm ─────────────────────────────────────────────────────────
void GestureDetBackend::post_process_palm(
  const cv::Mat & image,
  float ratio, cv::Point2i pad,
  float conf_thresh, float nms_thresh,
  std::vector<PalmObject> & palms)
{
  palms.clear();
  if (palm_ctx_.output_num < 2) { return; }

  // Identify box output (PALM_NUM_BOXES * PALM_NUM_COORDS) and score output (PALM_NUM_BOXES)
  int box_idx = -1, score_idx = -1;
  for (int i = 0; i < palm_ctx_.output_num; ++i) {
    const size_t elems = tensor_element_count(palm_ctx_.outs_attr[static_cast<size_t>(i)]);
    if (elems == static_cast<size_t>(PALM_NUM_BOXES * PALM_NUM_COORDS)) { box_idx   = i; }
    else if (elems == static_cast<size_t>(PALM_NUM_BOXES))              { score_idx = i; }
  }
  if (box_idx < 0 || score_idx < 0) { return; }

  const float * boxes_raw  = output_to_fp32(palm_ctx_, box_idx);
  const float * scores_raw = output_to_fp32(palm_ctx_, score_idx);
  if (!boxes_raw || !scores_raw) { return; }

  const int tw = (palm_ctx_.input_width  > 0) ? palm_ctx_.input_width  : config_.input_width;
  const int th = (palm_ctx_.input_height > 0) ? palm_ctx_.input_height : config_.input_height;
  if (ratio <= 0.0F) { return; }
  const float scale_to_origin = static_cast<float>(std::max(image.cols, image.rows));
  const float pad_bias_x = static_cast<float>(pad.x) / ratio;
  const float pad_bias_y = static_cast<float>(pad.y) / ratio;

  std::vector<cv::Rect2f> boxes;
  std::vector<float>      scores;
  std::vector<PalmObject> candidates;

  for (int i = 0; i < PALM_NUM_BOXES; ++i) {
    const float score = sigmoid(scores_raw[i]);
    if (score < conf_thresh) { continue; }

    const float * b  = boxes_raw + i * PALM_NUM_COORDS;
    const float ax   = kPalmAnchors[i].x;
    const float ay   = kPalmAnchors[i].y;

    const float dx = b[0] / static_cast<float>(tw);
    const float dy = b[1] / static_cast<float>(th);
    const float dw = b[2] / static_cast<float>(tw);
    const float dh = b[3] / static_cast<float>(th);
    const float x1 = (dx - dw * 0.5F + ax) * scale_to_origin - pad_bias_x;
    const float y1 = (dy - dh * 0.5F + ay) * scale_to_origin - pad_bias_y;
    const float x2 = (dx + dw * 0.5F + ax) * scale_to_origin - pad_bias_x;
    const float y2 = (dy + dh * 0.5F + ay) * scale_to_origin - pad_bias_y;
    if (x2 <= x1 || y2 <= y1) { continue; }

    PalmObject palm;
    palm.score   = score;
    palm.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

    // Decode 7 keypoints
    for (int k = 0; k < PALM_NUM_KEYPOINTS; ++k) {
      const float keypoint_x = b[4 + k * 2] / static_cast<float>(tw);
      const float keypoint_y = b[4 + k * 2 + 1] / static_cast<float>(th);
      palm.kpts[static_cast<size_t>(k)].x =
        (keypoint_x + ax) * scale_to_origin - pad_bias_x;
      palm.kpts[static_cast<size_t>(k)].y =
        (keypoint_y + ay) * scale_to_origin - pad_bias_y;
    }

    boxes.push_back(cv::Rect2f(palm.box.x, palm.box.y, palm.box.width, palm.box.height));
    scores.push_back(score);
    candidates.push_back(palm);
  }

  std::vector<int> picked;
  palm_nms(boxes, scores, picked, nms_thresh);

  palms.reserve(picked.size());
  for (int idx : picked) {
    palms.push_back(candidates[static_cast<size_t>(idx)]);
  }
}

// ── preprocess_landmark ───────────────────────────────────────────────────────
bool GestureDetBackend::preprocess_landmark(
  const cv::Mat & image, const PalmObject & palm,
  cv::Mat & out_input, cv::Rect2f & rotated_bbox,
  float & angle, cv::Matx23f & rot_mat, cv::Point2f & pad_bias)
{
  const int tw = (landmark_ctx_.input_width  > 0) ? landmark_ctx_.input_width  : 224;
  const int th = (landmark_ctx_.input_height > 0) ? landmark_ctx_.input_height : 224;

  cv::Mat rotation_crop;
  cv::Rect2f palm_bbox_in_original;
  if (!crop_and_pad_from_palm(
      image, palm.box, true, rotation_crop, palm_bbox_in_original, pad_bias))
  {
    return false;
  }

  cv::Mat rotation_crop_rgb;
  cv::cvtColor(rotation_crop, rotation_crop_rgb, cv::COLOR_BGR2RGB);

  cv::Rect2f local_palm_bbox = palm_bbox_in_original;
  local_palm_bbox.x -= pad_bias.x;
  local_palm_bbox.y -= pad_bias.y;

  std::array<cv::Point2f, PALM_NUM_KEYPOINTS> local_keypoints;
  for (int i = 0; i < PALM_NUM_KEYPOINTS; ++i) {
    local_keypoints[static_cast<size_t>(i)] = cv::Point2f(
      palm.kpts[static_cast<size_t>(i)].x - pad_bias.x,
      palm.kpts[static_cast<size_t>(i)].y - pad_bias.y);
  }

  const cv::Point2f p1 = local_keypoints[PALM_BASE_KEYPOINT];
  const cv::Point2f p2 = local_keypoints[MIDDLE_FINGER_BASE_KEYPOINT];
  float radians = static_cast<float>(CV_PI / 2.0) -
    std::atan2(-(p2.y - p1.y), p2.x - p1.x);
  radians -= static_cast<float>(2.0 * CV_PI) * std::floor(
    (radians + static_cast<float>(CV_PI)) / static_cast<float>(2.0 * CV_PI));
  angle = radians * 180.0F / static_cast<float>(CV_PI);

  const cv::Point2f rotation_center(
    local_palm_bbox.x + local_palm_bbox.width * 0.5F,
    local_palm_bbox.y + local_palm_bbox.height * 0.5F);
  rot_mat = rotation_matrix_2d(rotation_center, angle, 1.0F);

  cv::Mat rotated_image;
  cv::warpAffine(
    rotation_crop_rgb, rotated_image, cv::Mat(rot_mat), rotation_crop_rgb.size());

  std::vector<cv::Point2f> rotated_keypoints;
  rotated_keypoints.reserve(PALM_NUM_KEYPOINTS);
  for (const auto & keypoint : local_keypoints) {
    rotated_keypoints.push_back(transform_point(rot_mat, keypoint));
  }
  const cv::Rect2f landmark_bbox = bbox_from_points(rotated_keypoints);

  cv::Mat landmark_crop;
  cv::Point2f unused_bias;
  if (!crop_and_pad_from_palm(
      rotated_image, landmark_bbox, false, landmark_crop, rotated_bbox, unused_bias))
  {
    return false;
  }

  cv::resize(landmark_crop, out_input, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
  if (!out_input.isContinuous()) { out_input = out_input.clone(); }
  return true;
}

// ── post_process_landmark ─────────────────────────────────────────────────────
bool GestureDetBackend::post_process_landmark(
  const PalmObject & /*palm*/,
  const cv::Rect2f & rotated_bbox, float angle,
  const cv::Matx23f & rot_mat, const cv::Point2f & pad_bias,
  float conf_thresh, HandResult & hand)
{
  if (landmark_ctx_.output_num != 3) { return false; }

  constexpr int lm_idx = 0;
  constexpr int score_idx = 1;
  constexpr int hand_idx = 2;

  const float * lm_raw = output_to_fp32(landmark_ctx_, lm_idx);
  if (!lm_raw) { return false; }

  // Handedness and score
  const float * s = output_to_fp32(landmark_ctx_, score_idx);
  if (!s) { return false; }
  hand.score = s[0];
  if (hand.score < conf_thresh) { return false; }

  const float * h = output_to_fp32(landmark_ctx_, hand_idx);
  if (!h) { return false; }
  hand.handedness = h[0];

  const int input_width = landmark_ctx_.input_width > 0 ? landmark_ctx_.input_width : 224;
  const int input_height = landmark_ctx_.input_height > 0 ? landmark_ctx_.input_height : 224;
  const float scale_x = rotated_bbox.width / static_cast<float>(input_width);
  const float scale_y = rotated_bbox.height / static_cast<float>(input_height);
  const float max_scale = std::max(scale_x, scale_y);

  std::array<cv::Point3f, HAND_NUM_LANDMARKS> local_landmarks;
  for (int i = 0; i < HAND_NUM_LANDMARKS; ++i) {
    local_landmarks[static_cast<size_t>(i)] = cv::Point3f(
      (lm_raw[i * 3] - static_cast<float>(input_width) * 0.5F) * max_scale,
      (lm_raw[i * 3 + 1] - static_cast<float>(input_height) * 0.5F) * max_scale,
      lm_raw[i * 3 + 2] * max_scale);
  }

  const cv::Matx23f coordinate_rotation =
    rotation_matrix_2d(cv::Point2f(0.0F, 0.0F), angle, 1.0F);
  std::array<cv::Point3f, HAND_NUM_LANDMARKS> rotated_landmarks;
  for (int i = 0; i < HAND_NUM_LANDMARKS; ++i) {
    const auto & point = local_landmarks[static_cast<size_t>(i)];
    rotated_landmarks[static_cast<size_t>(i)] = cv::Point3f(
      point.x * coordinate_rotation(0, 0) + point.y * coordinate_rotation(1, 0),
      point.x * coordinate_rotation(0, 1) + point.y * coordinate_rotation(1, 1),
      point.z);
  }

  const float r00 = rot_mat(0, 0);
  const float r01 = rot_mat(0, 1);
  const float r10 = rot_mat(1, 0);
  const float r11 = rot_mat(1, 1);
  const float tx = rot_mat(0, 2);
  const float ty = rot_mat(1, 2);
  const float inverse_translation_x = -(r00 * tx + r10 * ty);
  const float inverse_translation_y = -(r01 * tx + r11 * ty);

  const cv::Point2f rotated_center(
    rotated_bbox.x + rotated_bbox.width * 0.5F,
    rotated_bbox.y + rotated_bbox.height * 0.5F);
  const cv::Point2f original_center(
    rotated_center.x * r00 + rotated_center.y * r10 + inverse_translation_x,
    rotated_center.x * r01 + rotated_center.y * r11 + inverse_translation_y);

  for (int i = 0; i < HAND_NUM_LANDMARKS; ++i) {
    const auto & point = rotated_landmarks[static_cast<size_t>(i)];
    hand.landmarks[static_cast<size_t>(i)] = cv::Point3f(
      point.x + original_center.x + pad_bias.x,
      point.y + original_center.y + pad_bias.y,
      point.z);
  }

  float min_x = hand.landmarks[0].x;
  float min_y = hand.landmarks[0].y;
  float max_x = hand.landmarks[0].x;
  float max_y = hand.landmarks[0].y;
  for (int i = 1; i < HAND_NUM_LANDMARKS; ++i) {
    const auto & point = hand.landmarks[static_cast<size_t>(i)];
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }

  cv::Rect2f hand_bbox(min_x, min_y, max_x - min_x, max_y - min_y);
  hand_bbox.x += HAND_BOX_SHIFT.x * hand_bbox.width;
  hand_bbox.y += HAND_BOX_SHIFT.y * hand_bbox.height;
  const cv::Point2f hand_center(
    hand_bbox.x + hand_bbox.width * 0.5F,
    hand_bbox.y + hand_bbox.height * 0.5F);
  const cv::Point2f hand_half(
    hand_bbox.width * HAND_BOX_ENLARGE * 0.5F,
    hand_bbox.height * HAND_BOX_ENLARGE * 0.5F);
  hand.box = cv::Rect2f(
    hand_center.x - hand_half.x, hand_center.y - hand_half.y,
    hand_half.x * 2.0F, hand_half.y * 2.0F);

  hand.gesture = recognize_gesture(hand.landmarks);
  return true;
}

// ── fill_input_tensor ─────────────────────────────────────────────────────────
void GestureDetBackend::fill_input_tensor(ModelContext & ctx, const cv::Mat & rgb, int idx)
{
  const taconn_inout_attr_t & attr = ctx.ins_attr[static_cast<size_t>(idx)];
  const size_t elems = tensor_element_count(attr);
  const bool is_nchw = (attr.dim_count == 4 && attr.dim_size[1] == 3);

  std::vector<uint8_t> buf;
  const uint8_t * src = rgb.data;

  if (is_nchw) {
    const int h = rgb.rows, w = rgb.cols;
    buf.resize(static_cast<size_t>(3 * h * w));
    std::vector<cv::Mat> chs; cv::split(rgb, chs);
    for (int c = 0; c < 3; ++c) {
      memcpy(buf.data() + static_cast<size_t>(c * h * w), chs[static_cast<size_t>(c)].data,
        static_cast<size_t>(h * w));
    }
    src = buf.data();
  }

  void * dst = ctx.input_tensors[idx].data;

  if (attr.quant_format == TACONN_QNT_TYPE_NONE) {
    if (attr.data_format == TACONN_DATA_FORMAT_FP32) {
      auto * out = static_cast<float *>(dst);
      for (size_t i = 0; i < elems; ++i) { out[i] = static_cast<float>(src[i]) / 255.0f; }
    } else if (attr.data_format == TACONN_DATA_FORMAT_FP16) {
      auto * out = static_cast<uint16_t *>(dst);
      for (size_t i = 0; i < elems; ++i) {
        out[i] = float_to_half(static_cast<float>(src[i]) / 255.0f);
      }
    } else {
      memcpy(dst, src, elems);
    }
  } else if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float scale = attr.quant_data.affine.tf_scale;
    const int32_t zp  = attr.quant_data.affine.tf_zero_point;
    if (scale == 0.0F) { return; }
    if (attr.data_format == TACONN_DATA_FORMAT_INT8) {
      auto * out = static_cast<int8_t *>(dst);
      for (size_t i = 0; i < elems; ++i) {
        int32_t q = static_cast<int32_t>(std::lrint(
          static_cast<float>(src[i]) / 255.0f / scale)) + zp;
        out[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
      }
    } else if (attr.data_format == TACONN_DATA_FORMAT_UINT8) {
      auto * out = static_cast<uint8_t *>(dst);
      for (size_t i = 0; i < elems; ++i) {
        int32_t q = static_cast<int32_t>(std::lrint(
          static_cast<float>(src[i]) / 255.0f / scale)) + zp;
        out[i] = static_cast<uint8_t>(std::max(0, std::min(255, q)));
      }
    }
  } else if (
    attr.quant_format == TACONN_QNT_TYPE_DFP &&
    attr.data_format == TACONN_DATA_FORMAT_INT8)
  {
    const int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
    auto * out = static_cast<int8_t *>(dst);
    for (size_t i = 0; i < elems; ++i) {
      const float normalized = static_cast<float>(src[i]) / 255.0F;
      const int32_t q = static_cast<int32_t>(
        std::lrint(std::ldexp(normalized, fixed_point_pos)));
      out[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
  }
}

// ── output_to_fp32 ────────────────────────────────────────────────────────────
const float * GestureDetBackend::output_to_fp32(ModelContext & ctx, int out_idx)
{
  if (out_idx < 0 || out_idx >= ctx.output_num) { return nullptr; }
  const taconn_inout_attr_t & attr = ctx.outs_attr[static_cast<size_t>(out_idx)];
  const size_t elems = tensor_element_count(attr);
  auto & cache = ctx.output_fp32_cache[static_cast<size_t>(out_idx)];
  if (cache.size() != elems) { cache.resize(elems); }

  const void * data = ctx.output_buffer[out_idx].data;
  for (size_t i = 0; i < elems; ++i) {
    cache[i] = decode_tensor_value(data, i, attr);
  }
  return cache.data();
}

// ── frame_to_bgr ──────────────────────────────────────────────────────────────
cv::Mat GestureDetBackend::frame_to_bgr(const InferenceFrame & frame)
{
  if (frame.data == nullptr || frame.data_size == 0) { return {}; }
  if (frame.format == PixelFormat::Bgr8) {
    return cv::Mat(frame.height, frame.width, CV_8UC3,
      const_cast<uint8_t *>(frame.data), static_cast<size_t>(frame.step)).clone();
  }
  if (frame.format == PixelFormat::Jpeg) {
    std::vector<uint8_t> buf(frame.data, frame.data + frame.data_size);
    return cv::imdecode(buf, cv::IMREAD_COLOR);
  }
  // NV12 → BGR via OpenCV
  cv::Mat nv12(frame.height + frame.height / 2, frame.width, CV_8UC1,
    const_cast<uint8_t *>(frame.data));
  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

}  // namespace topsbot_gesture_det

PLUGINLIB_EXPORT_CLASS(
  topsbot_gesture_det::GestureDetBackend,
  topsbot_ai_infer::IInferenceBackend)
