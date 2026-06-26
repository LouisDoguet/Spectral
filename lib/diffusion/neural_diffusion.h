#ifndef NEURAL_DIFFUSION_H
#define NEURAL_DIFFUSION_H

#include "../../neural/network.h"
#include "../space/mesh.h"
#include "diffusion.h"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace diff {

/**
 * @class NeuralNetwork
 * @brief Diffusion driven by a trained neural network.
 *
 * The network predicts optimal alpha values for the hybrid DGSEM solver.
 * Takes nodal density/energy values and outputs alpha [0,1] for blending
 * between high-order and low-order (FV) flux computations.
 *
 * For hybrid DGSEM:
 *   R = alpha * R_FV + (1-alpha) * R_DG
 *
 * Input:  Concatenated nodal values [rho_elem0, ..., rho_elemN]
 * Output: Per-element alpha values [alpha_elem0, ..., alpha_elemN]
 */
class NeuralNetwork : public _Diffusion {
public:
  /**
   * Load a trained neural network model.
   *
   * @param model_path Path to the saved .nn model file
   * @param normalizer_path Path to the .norm file with mean/std
   *                        (optional; if empty, no normalization)
   */
  NeuralNetwork(const std::string &model_path,
                const std::string &normalizer_path = "")
      : _Diffusion("NEURAL_NETWORK"), model_path(model_path),
        normalizer_path(normalizer_path), network(nullptr) {
    loadModel();
    if (!normalizer_path.empty()) {
      loadNormalizer();
    }
  }

  /**
   * Apply neural network to compute alpha field.
   * Predicts per-element alpha based on nodal density values.
   */
  void apply(mesh::Mesh *mesh) override;

private:
  std::string model_path;
  std::string normalizer_path;
  std::unique_ptr<Network> network;

  // Normalization parameters
  double norm_mean = 0.0;
  double norm_std = 1.0;
  bool has_normalizer = false;

  // Temporary buffers
  std::vector<double> input_buffer;  // Concatenated nodal values
  std::vector<double> output_buffer; // Per-element alpha values

  /**
   * Load the trained network model from file.
   */
  void loadModel() {
    try {
      auto loss_fn = std::make_shared<LFUN::MSE>();
      network = std::make_unique<Network>(Network::load(model_path, loss_fn));
      std::cout << "✓ Neural network loaded from: " << model_path << std::endl;
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to load neural network model: " +
                               std::string(e.what()));
    }
  }

  /**
   * Load normalization parameters from file.
   */
  void loadNormalizer() {
    std::ifstream f(normalizer_path, std::ios::binary);
    if (!f) {
      throw std::runtime_error("Cannot open normalizer file: " +
                               normalizer_path);
    }

    f.read(reinterpret_cast<char *>(&norm_mean), sizeof(double));
    f.read(reinterpret_cast<char *>(&norm_std), sizeof(double));
    f.close();

    has_normalizer = true;

    std::cout << "✓ Normalizer loaded from: " << normalizer_path << std::endl;
    std::cout << "  Mean: " << norm_mean << ", Std: " << norm_std << std::endl;

    if (norm_std < 1e-10) {
      throw std::runtime_error("Invalid normalizer: std is too small");
    }
  }

  /**
   * Normalize input data using stored parameters.
   */
  double normalize(double value) const {
    if (!has_normalizer)
      return value;
    return (value - norm_mean) / norm_std;
  }
};

} // namespace diff

#endif // NEURAL_DIFFUSION_H
