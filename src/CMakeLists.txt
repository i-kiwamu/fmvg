# Shared library
add_library(sequoiaortho SHARED
    preparation.cc
    matcher.cc
)

# Options
set(CMAKE_MACOSX_RPATH 1)
find_package(OpenCV REQUIRED)
find_package(Exiv2 REQUIRED)
target_compile_features(sequoiaortho PUBLIC cxx_std_11)
target_include_directories(sequoiaortho INTERFACE
    ./
    ${OpenCV_INCLUDE_DIRS}
    ${Exiv2_INCLUDE_DIRS}
)
target_link_libraries(sequoiaortho ${OpenCV_LIBS} exiv2lib)
