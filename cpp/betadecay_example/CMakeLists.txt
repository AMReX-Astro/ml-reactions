cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(AMReX REQUIRED)

find_package(Torch REQUIRED)

add_executable(main main.cpp)

# properties
# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(main PUBLIC cxx_std_14)
set_target_properties(main PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(main AMReX::amrex )

target_link_libraries(main "${TORCH_LIBRARIES}")
