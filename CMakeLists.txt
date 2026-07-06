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
set(time "lib/time/solver.cpp" "lib/time/rk4.cpp" "lib/time/hybrid_solver.cpp" "lib/time/hybrid_alpha.cpp")
set(diff "lib/diffusion/diffusion.cpp" "lib/diffusion/neural_diffusion.cpp")
set(sensor "lib/sensor/persson_peraire.cpp" "lib/sensor/div_laplacian.cpp")
set(post   "lib/post/vtu_exporter.cpp")
# 1D benchmark test cases (Sod + Lax/Shu-Osher/Woodward-Colella)
set(SOD "lib/SOD.cpp" "lib/lax.cpp" "lib/shu_osher.cpp" "lib/woodward_colella.cpp")
set(neural_core "neural/activation.cpp" "neural/layer.cpp" "neural/loss_function.cpp")
set(neural_test "neural/test_mnist.cpp")
set(neural_train "neural/train_alpha_predictor.cpp")

# Main spectral solver (includes neural network integration)
add_executable(spectral "main.cpp" ${elem} ${phy} ${math} ${time} ${base} ${diff} ${SOD} ${sensor} ${post} ${neural_core})

# MNIST test executable
add_executable(neural_test "neural/test_mnist.cpp" ${neural_core})

# Neural network training executable (RK4 artificial-diffusion alpha)
add_executable(train_alpha_predictor "neural/train_alpha_predictor.cpp" ${neural_core})

# Hybrid-DGSEM alpha policy training (one-step reward RL). Needs the full
# physics stack because it runs real hybrid-DGSEM trajectories and trial steps.
add_executable(train_hybrid_alpha "neural/train_hybrid_alpha.cpp"
    ${elem} ${phy} ${math} ${time} ${base} ${diff} ${SOD} ${sensor} ${post} ${neural_core})


target_link_libraries(spectral ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(neural_test ${BLAS_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(train_alpha_predictor ${BLAS_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(train_hybrid_alpha ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})


# GCC < 9 needs an explicit link for std::filesystem
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(spectral stdc++fs)
endif()
