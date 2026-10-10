// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0
//
// Standalone CLI for testing gesture detection without ROS.
// Usage: gesture_det_file_cli --palm <model.nb> --landmark <model.nb> --input <image_or_dir> [--conf 0.5] [--nms 0.3]

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "topsbot_gesture_det/gesture_det_backend.hpp"
#include "topsbot_ai_infer/inference_backend.hpp"

static void print_usage(const char * prog)
{
  std::cerr << "Usage: " << prog
            << " --palm <palm.nb> --landmark <landmark.nb>"
            << " --input <image_or_dir>"
            << " [--conf 0.5] [--nms 0.3] [--lm-conf 0.5]"
            << " [--output <dir>]\n";
}

int main(int argc, char ** argv)
{
  std::string palm_model, landmark_model, input_path, output_dir;
  float conf_thresh = 0.5F;
  float nms_thresh  = 0.3F;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--palm" || arg == "-p") && i + 1 < argc) {
      palm_model = argv[++i];
    } else if ((arg == "--landmark" || arg == "-l") && i + 1 < argc) {
      landmark_model = argv[++i];
    } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--conf" && i + 1 < argc) {
      conf_thresh = std::stof(argv[++i]);
    } else if (arg == "--nms" && i + 1 < argc) {
      nms_thresh = std::stof(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (palm_model.empty() || landmark_model.empty() || input_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  topsbot_gesture_det::GestureDetBackend detector;

  // Build a ModelConfig that matches our backend's expectations.
  topsbot_ai_infer::ModelConfig cfg;
  cfg.backend = "gesture_det";
  cfg.model_name = "gesture_det";
  cfg.model_type = "palm_hand_landmark";
  cfg.model_path = palm_model;
  cfg.confidence_threshold = conf_thresh;
  cfg.nms_threshold = nms_thresh;
  // Convention: landmark model path is stored in class_names[0].
  cfg.class_names.push_back(landmark_model);

  std::string err;
  if (!detector.initialize(cfg, err)) {
    std::cerr << "Failed to initialize detector: " << err << "\n";
    return 1;
  }

  // Collect image paths
  std::vector<std::string> images;
  namespace fs = std::filesystem;
  if (fs::is_directory(input_path)) {
    for (const auto & entry : fs::directory_iterator(input_path)) {
      if (!entry.is_regular_file()) { continue; }
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        images.push_back(entry.path().string());
      }
    }
    std::sort(images.begin(), images.end());
  } else {
    images.push_back(input_path);
  }

  if (!output_dir.empty()) {
    fs::create_directories(output_dir);
  }

  for (const auto & img_path : images) {
    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) {
      std::cerr << "Failed to read: " << img_path << "\n";
      continue;
    }

    // Wrap the BGR Mat into a framework InferenceFrame.
    topsbot_ai_infer::InferenceFrame frame;
    frame.owner     = nullptr;  // CLI owns the Mat locally
    frame.data      = bgr.data;
    frame.data_size = static_cast<std::size_t>(bgr.total() * bgr.elemSize());
    frame.width     = bgr.cols;
    frame.height    = bgr.rows;
    frame.step      = static_cast<int>(bgr.step);
    frame.format    = topsbot_ai_infer::PixelFormat::Bgr8;

    topsbot_ai_infer::InferenceResult result;
    try {
      result = detector.infer(frame);
    } catch (const std::exception & e) {
      std::cerr << "infer() threw on " << img_path << ": " << e.what() << "\n";
      continue;
    }
    if (!result.success) {
      std::cerr << "infer() failed on " << img_path << ": " << result.error << "\n";
      continue;
    }

    const auto & hands = result.frame.objects;
    std::cout << "[" << img_path << "] detected " << hands.size() << " hand(s)\n";
    for (size_t i = 0; i < hands.size(); ++i) {
      const auto & h = hands[i];
      std::string gesture = "(none)";
      float gesture_conf = 0.0F;
      std::string handedness = "(unknown)";
      for (const auto & attr : h.attributes) {
        if (attr.attr_id == 0) { gesture = attr.label; gesture_conf = attr.confidence; }
        else if (attr.attr_id == 1) { handedness = attr.label; }
      }
      std::cout << "  hand[" << i << "] gesture=" << gesture
                << " score=" << gesture_conf
                << " handedness=" << handedness
                << " box=[" << h.detector_bbox.x << "," << h.detector_bbox.y
                << "," << h.detector_bbox.x + h.detector_bbox.width
                << "," << h.detector_bbox.y + h.detector_bbox.height << "]\n";
    }

    // Optional: draw landmarks onto the image and save
    if (!output_dir.empty()) {
      cv::Mat out = bgr.clone();
      for (const auto & h : hands) {
        cv::rectangle(out,
          cv::Rect(static_cast<int>(h.detector_bbox.x), static_cast<int>(h.detector_bbox.y),
                   static_cast<int>(h.detector_bbox.width), static_cast<int>(h.detector_bbox.height)),
          cv::Scalar(0, 255, 0), 2);
        for (const auto & meta : h.user_metas) {
          auto lm = std::dynamic_pointer_cast<topsbot_ai_infer::LandmarkMeta>(meta);
          if (!lm) { continue; }
          // Draw mediapipe hand skeleton (21 keypoints, parent -> child
          // edges). The index list mirrors the canonical MediaPipe graph.
          static constexpr std::array<std::pair<int, int>, 21> edges = {{
            {0, 1}, {1, 2}, {2, 3}, {3, 4},            // thumb
            {0, 5}, {5, 6}, {6, 7}, {7, 8},            // index
            {5, 9}, {9, 10}, {10, 11}, {11, 12},       // middle
            {9, 13}, {13, 14}, {14, 15}, {15, 16},     // ring
            {13, 17}, {17, 18}, {18, 19}, {19, 20},    // pinky
            {0, 17},                                  // palm base
          }};
          for (const auto & edge : edges) {
            const auto & a = lm->points[static_cast<size_t>(edge.first)];
            const auto & b = lm->points[static_cast<size_t>(edge.second)];
            if ((a.x == 0.0F && a.y == 0.0F) || (b.x == 0.0F && b.y == 0.0F)) { continue; }
            cv::line(out,
                cv::Point(static_cast<int>(a.x), static_cast<int>(a.y)),
                cv::Point(static_cast<int>(b.x), static_cast<int>(b.y)),
                cv::Scalar(255, 0, 0), 2);
          }
          for (const auto & p : lm->points) {
            // mediapipe hand landmarks don't have a real confidence per
            // point; presence in lm->points means the backend has placed it.
            // (p.z carries a depth value, not a score, so it is intentionally
            // ignored here.)
            if (p.x != 0.0F || p.y != 0.0F) {
              cv::circle(out, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)),
                         3, cv::Scalar(0, 0, 255), -1);
            }
          }
        }
      }
      fs::path out_path = fs::path(output_dir) / fs::path(img_path).filename();
      cv::imwrite(out_path.string(), out);
    }
  }

  return 0;
}
