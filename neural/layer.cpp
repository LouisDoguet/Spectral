#include "layer.h"
#include "tensor.h"
#include <cstddef>

TENSOR::Tensor LAYER::_Layer::forward(const TENSOR::Tensor &input) {
  this->input_cache = input;
  /// LINEAR TRANSFORMATION
  this->preactivation_cache = (input * this->weights).add_bias(this->bias);
  TENSOR::Tensor output = preactivation_cache;
  /// ACTIVATION
  activation_function->apply(output);
  this->postactivation_cache = output;
  return output;
}

TENSOR::Tensor LAYER::_Layer::backward(const TENSOR::Tensor &grad_output) {
  TENSOR::Tensor act_grad =
      activation_function->gradient(this->postactivation_cache);
  TENSOR::Tensor dJ_dZ(grad_output.n_rows, grad_output.n_cols);
  for (size_t i = 0; i < grad_output.n_cols * grad_output.n_rows; ++i)
    dJ_dZ.setData(i, grad_output.readData()[i] * act_grad.getData()[i]);

  this->grad_weights = this->input_cache.T() * dJ_dZ;
  this->grad_bias = dJ_dZ.sum_rows();
  return dJ_dZ * this->weights.T();
}

void LAYER::_Layer::update(double learning_rate) {
  this->weights = this->weights - (this->grad_weights * learning_rate);
  this->bias = this->bias - (this->grad_bias * learning_rate);
};

/**
TENSOR::Tensor LAYER::SoftMax::backward(const TENSOR::Tensor &y_onehot) {
  TENSOR::Tensor grad = postactivation_cache - y_onehot;
  return _Layer::backward(grad);
}
*/
