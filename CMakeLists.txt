
cmake_minimum_required(VERSION 3.12)
project(mc_option_omp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(mc_option src/main.cpp)
target_link_libraries(mc_option PUBLIC OpenMP::OpenMP_CXX)
