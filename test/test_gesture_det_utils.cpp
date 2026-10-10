// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "topsbot_gesture_det/gesture_det_utils.hpp"

using namespace topsbot_gesture_det;

TEST(GestureDetUtils, SigmoidBounds)
{
  EXPECT_NEAR(sigmoid(0.0F), 0.5F, 1e-5F);
  EXPECT_NEAR(sigmoid(100.0F), 1.0F, 1e-5F);
  EXPECT_NEAR(sigmoid(-100.0F), 0.0F, 1e-5F);
}

TEST(GestureDetUtils, FloatHalfRoundtrip)
{
  const float vals[] = {0.0F, 1.0F, -1.0F, 0.5F, 3.14F};
  for (float v : vals) {
    float rt = half_to_float(float_to_half(v));
    EXPECT_NEAR(rt, v, std::abs(v) * 0.01F + 1e-3F);
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
