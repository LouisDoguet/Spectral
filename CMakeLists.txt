cmake_minimum_required(VERSION 3.10)
project(spectral)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

set(elem "lib/space/element.cpp" "lib/space/mesh.cpp" "lib/boundary_conditions/boundary_conditions.cpp")
set(phy  "lib/phy/physics.cpp" "lib/phy/entropy_flux.cpp")
set(base "lib/base/gll.cpp" "lib/base/base.cpp" "lib/base/radial_base_function.cpp")
set(math "lib/math/math.cpp")
set(time "lib/time/solver.cpp" "lib/time/rk4.cpp" "lib/time/hybrid_solver.cpp")
set(diff "lib/diffusion/diffusion.cpp")
set(sensor "lib/sensor/persson_peraire.cpp" "lib/sensor/div_laplacian.cpp")
set(post   "lib/post/vtu_exporter.cpp")
# 1D benchmark test cases (Sod + Lax/Shu-Osher/Woodward-Colella)
set(SOD "lib/SOD.cpp" "lib/lax.cpp" "lib/shu_osher.cpp" "lib/woodward_colella.cpp")
# C++ inference of the JAX/equinox alpha policy (loads a .nnx from nn/export/export.py)
set(equinox "lib/equinox/conv1d.cpp" "lib/equinox/resblock.cpp" "lib/equinox/alpha_net.cpp")

# Main spectral solver (hybrid DGSEM; alpha from Persson-Peraire or the exported NN)
add_executable(spectral "main.cpp" ${elem} ${phy} ${math} ${time} ${base} ${diff} ${SOD} ${sensor} ${post} ${equinox})

target_link_libraries(spectral ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})


# GCC < 9 needs an explicit link for std::filesystem
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(spectral stdc++fs)
endif()

# Optional verification harness for the JAX -> C++ alpha export (not built by
# default). Pair with nn/export/parity_check.py; see NN_EXPORT_MIGRATION.md.
#   cmake --build build --target equinox_parity
add_executable(equinox_parity EXCLUDE_FROM_ALL "nn/export/equinox_parity.cpp"
    ${elem} ${phy} ${math} ${base} ${diff} ${SOD} ${sensor} ${post} ${equinox})
target_link_libraries(equinox_parity ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(equinox_parity stdc++fs)
endif()
