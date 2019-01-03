cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch-test)

find_package(Torch REQUIRED)

add_executable(libtorch-test cpptest.cpp)
message("torch libraries variable: ${TORCH_LIBRARIES}")
target_link_libraries(libtorch-test "${TORCH_LIBRARIES}")
set_property(TARGET libtorch-test PROPERTY CXX_STANDARD 11)