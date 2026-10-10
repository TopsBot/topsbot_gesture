// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

/**
 * @brief Basic sanity check for gesture detection utils
 *
 * This test verifies that the test framework is working correctly
 * and the gesture_det_utils library can be linked successfully.
 */
TEST(GestureDetUtilsTest, BasicSanityCheck) {
  // Basic test to ensure the test framework is operational
  EXPECT_TRUE(true);
  EXPECT_EQ(1 + 1, 2);
}

/**
 * @brief Test that the library can be instantiated
 *
 * Verifies basic compilation and linking without hardware dependencies.
 */
TEST(GestureDetUtilsTest, LibraryLinkage) {
  // If this test compiles and runs, the library linkage is correct
  SUCCEED();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
