# Produce test executable file
add_executable(a.out main.cc)

# Link sequoiaortho library when test is compiled
target_compile_options(a.out PUBLIC -O2 -Wall)
target_compile_features(a.out PUBLIC cxx_std_11)
target_link_libraries(a.out sequoiaortho)
