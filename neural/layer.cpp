/**
 * @file layer.cpp
 * @brief Contains method definitions for Layer object. `_Layer` is the dummy class for all the developped layer types
 */

#include "layer.h"
#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

void LAYER::_Layer::serialize(std::ofstream &f) const {
  uint8_t act_id = activation_function->typeId();
  uint64_t in = in_features, out = out_features;
  f.write(reinterpret_cast<const char *>(&act_id), 1);
  f.write(reinterpret_cast<const char *>(&in), 8);
  f.write(reinterpret_cast<const char *>(&out), 8);
  const auto &w = weights.readData();
  const auto &b = bias.readData();
  f.write(reinterpret_cast<const char *>(w.data()), (std::streamsize)(w.size() * sizeof(double)));
  f.write(reinterpret_cast<const char *>(b.data()), (std::streamsize)(b.size() * sizeof(double)));
}

void LAYER::_Layer::loadWeightsBias(std::ifstream &f) {
    std::vector<double> w(in_features * out_features);
    std::vector<double> b(out_features);
    f.read(reinterpret_cast<char *>(w.data()), (std::streamsize)(w.size() * sizeof(double)));
    f.read(reinterpret_cast<char *>(b.data()), (std::streamsize)(b.size() * sizeof(double)));
    weights.setData(w);
    bias.setData(b);
}

/**
TENSOR::Tensor LAYER::SoftMax::backward(const TENSOR::Tensor &y_onehot) {
  TENSOR::Tensor grad = postactivation_cache - y_onehot;
  return _Layer::backward(grad);
}
*/
