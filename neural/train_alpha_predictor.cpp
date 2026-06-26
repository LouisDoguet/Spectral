/**
 * @file train_alpha_predictor.cpp
 * @brief Train a neural network to predict optimal alpha values for hybrid
 * DGSEM
 *
 * This program:
 * 1. Generates synthetic training data (smooth vs shock regions)
 * 2. Normalizes the data
 * 3. Trains a neural network to predict alpha from solution gradients
 * 4. Saves the trained model
 */

#include "activation.h"
#include "container.h"
#include "layer.h"
#include "loss_function.h"
#include "network.h"
#include "tensor.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// ============================================================================
// TRAINING DATA GENERATOR
// ============================================================================

class AlphaDataGenerator {
public:
  AlphaDataGenerator(int num_elements = 20, int nodes_per_elem = 10)
      : num_elements(num_elements), nodes_per_elem(nodes_per_elem),
        total_nodes(num_elements * nodes_per_elem) {}

  /**
   * Generate smooth solution (no shocks)
   */
  std::vector<double> generateSmoothSolution(std::mt19937 &rng) {
    std::vector<double> solution(total_nodes);

    for (int i = 0; i < total_nodes; ++i) {
      double x = (double)i / total_nodes;
      // Smooth: sin + cos
      solution[i] = std::sin(2.0 * M_PI * x) + 0.5 * std::cos(4.0 * M_PI * x);
    }

    return solution;
  }

  /**
   * Generate shock solution with discontinuity
   */
  std::vector<double> generateShockSolution(std::mt19937 &rng,
                                            double shock_pos = 0.5) {
    std::vector<double> solution(total_nodes);
    double shock_width = 0.05;

    for (int i = 0; i < total_nodes; ++i) {
      double x = (double)i / total_nodes;

      // Step function
      if (x < shock_pos) {
        solution[i] = 1.0;
      } else {
        solution[i] = 0.25;
      }

      // Smooth transition zone
      if (std::abs(x - shock_pos) < shock_width) {
        solution[i] = 0.625; // midpoint
      }
    }

    return solution;
  }

  /**
   * Compute ground truth alpha from solution gradients
   * - Low gradient (smooth): alpha ≈ 0
   * - High gradient (shock): alpha ≈ 1
   */
  std::vector<double>
  computeGroundTruthAlpha(const std::vector<double> &solution) {
    std::vector<double> alpha(total_nodes, 0.0);

    // Compute gradient using finite differences
    std::vector<double> grad(total_nodes);
    for (int i = 0; i < total_nodes; ++i) {
      int im = (i - 1 + total_nodes) % total_nodes;
      int ip = (i + 1) % total_nodes;
      grad[i] = std::abs(solution[ip] - solution[im]) / 2.0;
    }

    // Find max gradient for normalization
    double grad_max = 0.0;
    for (double g : grad) {
      grad_max = std::max(grad_max, g);
    }
    grad_max = std::max(grad_max, 1e-8);

    // Threshold and smooth
    double smooth_threshold = 0.1;
    for (int i = 0; i < total_nodes; ++i) {
      double grad_norm = grad[i] / grad_max;
      alpha[i] = (grad_norm > smooth_threshold) ? 1.0 : 0.0;
    }

    // Apply Gaussian smoothing
    std::vector<double> alpha_smooth(total_nodes, 0.0);
    double sigma = 1.5;
    for (int i = 0; i < total_nodes; ++i) {
      for (int j = 0; j < total_nodes; ++j) {
        double dist = std::abs(i - j);
        if (dist > total_nodes / 2) {
          dist = total_nodes - dist; // periodic
        }
        double weight = std::exp(-dist * dist / (2.0 * sigma * sigma));
        alpha_smooth[i] += alpha[j] * weight;
      }
      alpha_smooth[i] /= std::sqrt(2.0 * M_PI * sigma * sigma);
      alpha_smooth[i] = std::max(0.0, std::min(1.0, alpha_smooth[i]));
    }

    return alpha_smooth;
  }

  /**
   * Generate complete training dataset
   */
  void generateDataset(TENSOR::Tensor &input_tensor,
                       TENSOR::Tensor &target_tensor, int num_smooth = 100,
                       int num_shocks = 100) {
    int total_samples = num_smooth + num_shocks;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_pos(0.2, 0.8);

    std::vector<double> all_inputs;
    std::vector<double> all_targets;

    std::cout << "Generating " << num_smooth << " smooth samples..."
              << std::endl;
    for (int i = 0; i < num_smooth; ++i) {
      auto solution = generateSmoothSolution(rng);
      auto alpha = computeGroundTruthAlpha(solution);

      all_inputs.insert(all_inputs.end(), solution.begin(), solution.end());
      all_targets.insert(all_targets.end(), alpha.begin(), alpha.end());

      if ((i + 1) % 25 == 0) {
        std::cout << "  ✓ " << (i + 1) << " smooth samples generated"
                  << std::endl;
      }
    }

    std::cout << "Generating " << num_shocks << " shock samples..."
              << std::endl;
    for (int i = 0; i < num_shocks; ++i) {
      double shock_pos = dist_pos(rng);
      auto solution = generateShockSolution(rng, shock_pos);
      auto alpha = computeGroundTruthAlpha(solution);

      all_inputs.insert(all_inputs.end(), solution.begin(), solution.end());
      all_targets.insert(all_targets.end(), alpha.begin(), alpha.end());

      if ((i + 1) % 25 == 0) {
        std::cout << "  ✓ " << (i + 1) << " shock samples generated"
                  << std::endl;
      }
    }

    // Create tensors
    input_tensor = TENSOR::Tensor(total_samples, total_nodes);
    target_tensor = TENSOR::Tensor(total_samples, total_nodes);

    input_tensor.setData(all_inputs);
    target_tensor.setData(all_targets);

    std::cout << "\n✓ Dataset generated: " << total_samples << " samples, "
              << total_nodes << " nodes per sample" << std::endl;
  }

  // Getters
  int getTotalNodes() const { return total_nodes; }

private:
  int num_elements;
  int nodes_per_elem;
  int total_nodes;
};

// ============================================================================
// DATA NORMALIZATION
// ============================================================================

class DataNormalizer {
public:
  void fit(const TENSOR::Tensor &data) {
    const auto &values = data.readData();

    // Compute mean
    mean = 0.0;
    for (double v : values) {
      mean += v;
    }
    mean /= values.size();

    // Compute standard deviation
    double variance = 0.0;
    for (double v : values) {
      double diff = v - mean;
      variance += diff * diff;
    }
    variance /= values.size();
    std_dev = std::sqrt(variance);
    std_dev = std::max(std_dev, 1e-8); // Avoid division by zero

    std::cout << "Normalizer fitted:" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std:  " << std_dev << std::endl;
  }

  TENSOR::Tensor normalize(const TENSOR::Tensor &data) const {
    TENSOR::Tensor result = data;
    auto &values = result.getData();

    for (double &v : values) {
      v = (v - mean) / std_dev;
    }

    return result;
  }

  double denormalize(double norm_val) const {
    return norm_val * std_dev + mean;
  }

  double getMean() const { return mean; }
  double getStdDev() const { return std_dev; }

private:
  double mean = 0.0;
  double std_dev = 1.0;
};

// ============================================================================
// MAIN TRAINING FUNCTION
// ============================================================================

int main(int argc, char *argv[]) {
  try {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ALPHA PREDICTOR NEURAL NETWORK TRAINING" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    // Configuration
    const int NUM_ELEMENTS = 50;
    const int NODES_PER_ELEMENT = 7;
    const int TOTAL_NODES = NUM_ELEMENTS * NODES_PER_ELEMENT;
    const int EPOCHS = 5000;
    const double LEARNING_RATE = 1e-3;
    const int NUM_SMOOTH = 500;
    const int NUM_SHOCKS = 500;

    // Parse command line args
    std::string model_name = "alpha_model.nn";
    if (argc > 1) {
      model_name = argv[1];
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Total nodes per sample: " << TOTAL_NODES << std::endl;
    std::cout << "  Smooth samples: " << NUM_SMOOTH << std::endl;
    std::cout << "  Shock samples: " << NUM_SHOCKS << std::endl;
    std::cout << "  Total samples: " << (NUM_SMOOTH + NUM_SHOCKS) << std::endl;
    std::cout << "  Epochs: " << EPOCHS << std::endl;
    std::cout << "  Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "  Output model: " << model_name << "\n" << std::endl;

    // ────────────────────────────────────────────────────────────────
    // 1. Generate data
    // ────────────────────────────────────────────────────────────────
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "STEP 1: GENERATING TRAINING DATA" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    AlphaDataGenerator gen(NUM_ELEMENTS, NODES_PER_ELEMENT);
    TENSOR::Tensor input_raw, target_raw;
    gen.generateDataset(input_raw, target_raw, NUM_SMOOTH, NUM_SHOCKS);

    // ────────────────────────────────────────────────────────────────
    // 2. Normalize data
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "STEP 2: NORMALIZING DATA" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    DataNormalizer normalizer;
    normalizer.fit(input_raw);

    TENSOR::Tensor input = normalizer.normalize(input_raw);
    TENSOR::Tensor target = target_raw; // Targets already in [0,1]

    std::cout << "\nInput statistics after normalization:" << std::endl;
    const auto &input_data = input.readData();
    double input_mean = 0.0, input_var = 0.0;
    for (double v : input_data)
      input_mean += v;
    input_mean /= input_data.size();
    for (double v : input_data)
      input_var += (v - input_mean) * (v - input_mean);
    input_var /= input_data.size();
    std::cout << "  Mean: " << input_mean << " (should be ≈ 0)" << std::endl;
    std::cout << "  Std:  " << std::sqrt(input_var) << " (should be ≈ 1)"
              << std::endl;

    // ────────────────────────────────────────────────────────────────
    // 3. Create network
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "STEP 3: CREATING NETWORK ARCHITECTURE" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    // Architecture: [total_nodes] -> [64] -> [64] -> [total_nodes]
    auto architecture = std::make_shared<CONT::Sequential>();
    architecture->add(std::make_shared<LAYER::ReLU>(TOTAL_NODES, 64));
    architecture->add(std::make_shared<LAYER::ReLU>(64, 64));
    architecture->add(std::make_shared<LAYER::Sigmoid>(64, TOTAL_NODES));

    std::cout << "Network layers:" << std::endl;
    std::cout << "  Layer 1: ReLU(" << TOTAL_NODES << " -> 64)" << std::endl;
    std::cout << "  Layer 2: ReLU(64 -> 64)" << std::endl;
    std::cout << "  Layer 3: Sigmoid(64 -> " << TOTAL_NODES << ")" << std::endl;

    // Create loss function
    auto loss_fn = std::make_shared<LFUN::MSE>();
    std::cout << "\nLoss function: Mean Squared Error (MSE)" << std::endl;

    // Create network
    Network network(architecture, loss_fn, LEARNING_RATE);
    std::cout << "\n✓ Network created and initialized" << std::endl;

    // ────────────────────────────────────────────────────────────────
    // 4. Train
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "STEP 4: TRAINING" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    network.train(input, target, EPOCHS);

    // ────────────────────────────────────────────────────────────────
    // 5. Save model
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "STEP 5: SAVING MODEL" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    network.save(model_name);
    std::cout << "✓ Model saved to: " << model_name << std::endl;

    // Save normalizer parameters
    std::string normalizer_file = model_name;
    normalizer_file.replace(normalizer_file.end() - 3, normalizer_file.end(),
                            ".norm");
    std::ofstream norm_file(normalizer_file, std::ios::binary);
    if (!norm_file) {
      throw std::runtime_error("Cannot open normalizer file for writing");
    }

    double mean = normalizer.getMean();
    double std_dev = normalizer.getStdDev();
    norm_file.write(reinterpret_cast<const char *>(&mean), sizeof(double));
    norm_file.write(reinterpret_cast<const char *>(&std_dev), sizeof(double));
    norm_file.close();

    std::cout << "✓ Normalizer saved to: " << normalizer_file << std::endl;

    // ────────────────────────────────────────────────────────────────
    // 6. Test on a few examples
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "STEP 6: VALIDATION ON TEST SAMPLES" << std::endl;
    std::cout << std::string(70, '-') << "\n" << std::endl;

    std::cout << "Testing on first 3 samples:" << std::endl;
    for (int i = 0; i < std::min(3, (int)input.n_rows); ++i) {
      // Extract sample
      TENSOR::Tensor sample(1, TOTAL_NODES);
      std::vector<double> sample_data(TOTAL_NODES);
      const auto &input_data = input.readData();
      for (int j = 0; j < TOTAL_NODES; ++j) {
        sample_data[j] = input_data[i * TOTAL_NODES + j];
      }
      sample.setData(sample_data);

      // Predict
      TENSOR::Tensor pred = network.predict(sample);
      const auto &pred_data = pred.readData();

      // Get target
      const auto &target_data = target.readData();

      // Compute error
      double mae = 0.0;
      for (int j = 0; j < TOTAL_NODES; ++j) {
        mae += std::abs(pred_data[j] - target_data[i * TOTAL_NODES + j]);
      }
      mae /= TOTAL_NODES;

      std::cout << "  Sample " << (i + 1) << ": MAE = " << mae << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "TRAINING COMPLETE!" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n✗ ERROR: " << e.what() << std::endl;
    return 1;
  }
}
