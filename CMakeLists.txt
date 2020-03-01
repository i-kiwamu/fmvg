# CMake version
cmake_minimum_required(VERSION 2.8)

# Project name & language
project(fmvg CXX)

set(CMAKE_CXX_FLAGS "-O1 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# add subdirectories
add_subdirectory(src)
add_subdirectory(test)
