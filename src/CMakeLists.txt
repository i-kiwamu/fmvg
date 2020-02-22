# Shared library
add_library(fmvg SHARED
    photo.cc
    photo_correction.cc
    matcher.cc
)

# Options
set(CMAKE_MACOSX_RPATH 1)
find_package(OpenCV REQUIRED)
find_package(Exiv2 REQUIRED)
target_compile_features(fmvg PUBLIC cxx_std_11)
target_include_directories(fmvg INTERFACE
    ./
    ${OpenCV_INCLUDE_DIRS}
    ${Exiv2_INCLUDE_DIRS}
)
target_link_libraries(fmvg ${OpenCV_LIBS} exiv2lib)
