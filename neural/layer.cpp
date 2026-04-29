#include "layer.h"

TENSOR::Tensor LAYER::Linear::forward(const TENSOR::Tensor& input) {
    this->input_cache = input;
    return (input * weights) + bias;
}

std::unique_ptr<LAYER::Layer> LAYER::Linear::generateLayerFrom(const size_t next_out_features) const {
        return std::make_unique<LAYER::Linear>(this->out_features, next_out_features);
}