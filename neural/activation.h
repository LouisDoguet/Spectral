#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <cstdint>
#include <string>

namespace ACTI {
class Activation {
public:
  Activation(std::string name) : name(name) {};
  virtual void apply(TENSOR::Tensor &tensor) = 0;
  virtual TENSOR::Tensor gradient(TENSOR::Tensor &tensor) = 0;
  const std::string& getName() const { return name; }
  virtual uint8_t typeId() const = 0;

protected:
  std::string name;
};

class ReLU : public Activation {
public:
  ReLU() : Activation("ReLU") {}
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 0; }
};

class SoftMax : public Activation {
public:
  SoftMax() : Activation("SoftMax") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 2; }
};

class Sigmoid : public Activation {
public:
  Sigmoid() : Activation("Sigmoid") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 1; }
};

class HyperbolicTangent : public Activation {
public:
  HyperbolicTangent() : Activation("HyperbolicTangent") {};
  void apply(TENSOR::Tensor &tensor) override;
  TENSOR::Tensor gradient(TENSOR::Tensor &tensor);
  uint8_t typeId() const override { return 3; }
};

} // namespace ACTI

#endif
