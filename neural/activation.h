#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <string>

namespace ACTI {
class Activation {
public:
  Activation(std::string name) : name(name) {};
  virtual void apply(TENSOR::Tensor &tensor) = 0;
  virtual void gradient(TENSOR::Tensor &tensor) = 0;

protected:
  std::string name;
};

class ReLU : public Activation {
public:
  ReLU() : Activation("ReLU") {}
  void apply(TENSOR::Tensor &tensor) override;
  void gradient(TENSOR::Tensor &tensor);
};

class SoftMax : public Activation {
public:
  SoftMax() : Activation("SoftMax") {};
  void apply(TENSOR::Tensor &tensor) override;
  void gradient(TENSOR::Tensor &tensor);
};

class Sigmoid : public Activation {
public:
  Sigmoid() : Activation("Sigmoid") {};
  void apply(TENSOR::Tensor &tensor) override;
  void gradient(TENSOR::Tensor &tensor);
};

class HyperbolicTangent : public Activation {
public:
  HyperbolicTangent() : Activation("HyperbolicTangent") {};
  void apply(TENSOR::Tensor &tensor) override;
  void gradient(TENSOR::Tensor &tensor);
};

} // namespace ACTI

#endif
