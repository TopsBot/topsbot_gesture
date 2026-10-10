cmake_minimum_required(VERSION 3.8)
project(topsbot_gesture_det)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(topsbot_ai_infer REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_SYSROOT)
  set(GESTURE_SYSTEM_ROOT "${CMAKE_SYSROOT}/usr/local")
else()
  set(GESTURE_SYSTEM_ROOT "/usr/local")
endif()
set(GESTURE_SYSTEM_INCLUDE "${GESTURE_SYSTEM_ROOT}/include")
set(GESTURE_SYSTEM_LIB "${GESTURE_SYSTEM_ROOT}/lib")

# ── utils (tensor helpers) ──────────────────────────────────────────────────
add_library(topsbot_gesture_det_utils SHARED
  src/gesture_det_utils.cpp
)
target_include_directories(topsbot_gesture_det_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${GESTURE_SYSTEM_INCLUDE}
)
ament_target_dependencies(topsbot_gesture_det_utils topsbot_ai_infer)

# ── backend plugin ─────────────────────────────────────────────────────────
add_library(topsbot_gesture_det_backend SHARED
  src/gesture_det_backend.cpp
)
target_include_directories(topsbot_gesture_det_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${GESTURE_SYSTEM_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(topsbot_gesture_det_backend PUBLIC
  ${GESTURE_SYSTEM_LIB}
)
target_link_libraries(topsbot_gesture_det_backend
  topsbot_gesture_det_utils
  ta_runtime
  ${OpenCV_LIBS}
)
ament_target_dependencies(topsbot_gesture_det_backend
  pluginlib
  topsbot_ai_infer
)

# ── CLI tool (file/camera offline test) ────────────────────────────────────
add_executable(gesture_det_file_cli
  src/gesture_det_file_cli.cpp
)
target_include_directories(gesture_det_file_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GESTURE_SYSTEM_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(gesture_det_file_cli PRIVATE ${GESTURE_SYSTEM_LIB})
target_link_libraries(gesture_det_file_cli
  topsbot_gesture_det_backend
  topsbot_gesture_det_utils
  ta_runtime
  ${OpenCV_LIBS}
)
ament_target_dependencies(gesture_det_file_cli
  ament_index_cpp
  topsbot_ai_infer
)

# ── pluginlib export ────────────────────────────────────────────────────────
pluginlib_export_plugin_description_file(topsbot_ai_infer gesture_det_plugins.xml)

# ── install ─────────────────────────────────────────────────────────────────
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS
    topsbot_gesture_det_utils
    topsbot_gesture_det_backend
    gesture_det_file_cli
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config launch scripts
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES gesture_det_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Basic unit test for gesture detection utils
  ament_add_gtest(test_gesture_utils
    test/test_gesture_utils.cpp
  )
  target_link_libraries(test_gesture_utils
    topsbot_gesture_det_utils
  )
  target_include_directories(test_gesture_utils PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
endif()

ament_export_include_directories(include)
ament_export_dependencies(pluginlib topsbot_ai_infer)

ament_package()
