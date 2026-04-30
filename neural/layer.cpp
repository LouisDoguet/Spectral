#include "layer.h"

TENSOR::Tensor LAYER::Linear::forward(const TENSOR::Tensor& input) {
    this->input_cache = input;
    return (input * weights).add_bias(bias);
}

TENSOR::Tensor LAYER::Linear::backward(const TENSOR::Tensor& grad_output) {
    grad_weights = input_cache.T() * grad_output;
    grad_bias = grad_output.sum_rows();
    TENSOR::Tensor grad_input = grad_output * weights.T();
    return grad_input;
}

std::unique_ptr<LAYER::Layer> LAYER::Linear::generateLayerFrom(const size_t next_out_features) const {
        return std::make_unique<LAYER::Linear>(this->out_features, next_out_features);
}