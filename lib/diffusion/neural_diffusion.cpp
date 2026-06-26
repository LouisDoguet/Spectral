#include "neural_diffusion.h"
#include "../../neural/network.h"
#include "../space/mesh.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace diff {

void NeuralNetwork::apply(mesh::Mesh *mesh) {
  if (!network) {
    throw std::runtime_error("Neural network not loaded");
  }

  int num_elements = mesh->getNumElements();
  int nodes_per_elem = mesh->getBasis()->getOrder() + 1;
  int total_nodes_expected = num_elements * nodes_per_elem;

  // Prepare input: concatenate nodal density values from all elements
  input_buffer.resize(total_nodes_expected, 0.0);

  // Extract density values (rho) from each element
  double *global_rho = mesh->getGlobalU1();
  for (int i = 0; i < total_nodes_expected; ++i) {
    input_buffer[i] = normalize(global_rho[i]);
  }

  // Create input tensor and run network
  TENSOR::Tensor input_tensor(1, total_nodes_expected);
  input_tensor.setData(input_buffer);

  TENSOR::Tensor output = network->predict(input_tensor);
  const auto &output_data = output.readData();

  // Reduce nodal predictions to per-element alpha by averaging
  output_buffer.resize(num_elements, 0.0);
  for (int elem_id = 0; elem_id < num_elements; ++elem_id) {
    double alpha_sum = 0.0;
    for (int node = 0; node < nodes_per_elem; ++node) {
      int global_idx = elem_id * nodes_per_elem + node;
      alpha_sum += output_data[global_idx];
    }
    output_buffer[elem_id] = alpha_sum / nodes_per_elem;
    // Clamp to [0, 1]
    output_buffer[elem_id] =
        std::max(0.0, std::min(1.0, output_buffer[elem_id]));
  }

  // Pass per-element alpha to mesh for hybrid residual computation
  // (mesh->computeHybridResidual expects per-element alpha array)
  mesh->computeHybridResidual(output_buffer.data());

  // Optionally log statistics
  if (num_elements > 0) {
    double alpha_min = output_buffer[0];
    double alpha_max = output_buffer[0];
    double alpha_mean = 0.0;

    for (double alpha : output_buffer) {
      alpha_min = std::min(alpha_min, alpha);
      alpha_max = std::max(alpha_max, alpha);
      alpha_mean += alpha;
    }
    alpha_mean /= num_elements;

    std::cout << "  Neural network alpha: min=" << alpha_min
              << " mean=" << alpha_mean << " max=" << alpha_max << std::endl;
  }
}

} // namespace diff
