#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <cstdint>
#include <string>

namespace ACTI {

/**
 * @brief Activation function dummy class
 */
class Activation {
public:
  /**
   * @brief Activation constructor
   * @param name Name of the activation function
   */
  Activation(std::string name) : name(name) {};

  /**
   * @brief Activate a tensor
   * @param tensor Tensor object adress
   * @return None
   */
  virtual void apply(TENSOR::Tensor &tensor) = 0;

  /**
   * @brief Computes the gradient of a tensor via the analytical solution of the AF differentiation
   * @param tensor Tensor object address
   * @return None
   */
  virtual TENSOR::Tensor gradient(TENSOR::Tensor &tensor) = 0;

  const std::string& getName() const { return name; }
  virtual uint8_t typeId() const = 0;

protected:
  std::string name;
};

/// @brief Rectified Linear Unit
class ReLU : public Activation {
public:
  ReLU() : Activation("ReLU") {}
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 0; }
};

/// @brief  Softmax
class SoftMax : public Activation {
public:
  SoftMax() : Activation("SoftMax") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 2; }
};

/// @brief Sigmoid
class Sigmoid : public Activation {
public:
  Sigmoid() : Activation("Sigmoid") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 1; }
};

/// @brief Hyperbolic Tangent
class HyperbolicTangent : public Activation {
public:
  HyperbolicTangent() : Activation("HyperbolicTangent") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 3; }
};

} // namespace ACTI

#endif
