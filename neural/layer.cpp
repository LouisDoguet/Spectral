#include "layer.h"

TENSOR::Tensor LAYER::Linear::forward(const TENSOR::Tensor& input) {
    return (input * weights) + bias;
}