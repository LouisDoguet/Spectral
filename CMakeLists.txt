cmake_minimum_required(VERSION 3.10)
project(spectral)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(BLAS REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

set(elem "lib/space/element.cpp" "lib/space/mesh.cpp")
set(phy  "lib/phy/physics.cpp")
set(base "lib/base/gll.cpp")
set(math "lib/math/math.cpp")
set(time "lib/time/rk4.cpp")
set(diff "lib/diffusion/diffusion.cpp")

add_executable(spectral "main.cpp" ${elem} ${phy} ${math} ${time} ${base} ${diff})
target_link_libraries(spectral ${BLAS_LIBRARIES} ${Boost_LIBRARIES})

# GCC < 9 needs an explicit link for std::filesystem
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(spectral stdc++fs)
endif()

# -----------------------------------------------------------------------
# Optional: ONNX Runtime (enable with -DWITH_ONNX=ON)
# Point to your installation via -DONNXRUNTIME_ROOT=<path> or the env var.
# -----------------------------------------------------------------------
option(WITH_ONNX "Enable ONNX Runtime for neural network inference" OFF)
if(WITH_ONNX)
    find_path(ORT_INCLUDE onnxruntime_cxx_api.h
              PATH_SUFFIXES onnxruntime
              HINTS $ENV{ONNXRUNTIME_ROOT}/include
              PATHS /usr/local/include /usr/include)
    find_library(ORT_LIB onnxruntime
                 HINTS $ENV{ONNXRUNTIME_ROOT}/lib
                 PATHS /usr/local/lib /usr/lib)
    if(NOT ORT_INCLUDE OR NOT ORT_LIB)
        message(FATAL_ERROR
            "ONNX Runtime not found.\n"
            "Install it and set ONNXRUNTIME_ROOT, or pass "
            "-DONNXRUNTIME_ROOT=<path> to cmake.")
    endif()
    target_include_directories(spectral PRIVATE ${ORT_INCLUDE})
    target_link_libraries(spectral ${ORT_LIB})
    target_compile_definitions(spectral PRIVATE WITH_ONNX)
    message(STATUS "ONNX Runtime: ${ORT_LIB}")
endif()
