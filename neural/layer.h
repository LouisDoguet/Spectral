#pragma once
#include "tensor.h"

namespace LAYER {

class Layer {
protected:
    const size_t in_features;
    const size_t out_features;
    TENSOR::Tensor weights;
    TENSOR::Tensor bias;

public:
    Layer(const size_t in, const size_t out) 
        : in_features(in), out_features(out),
          weights(in, out), bias(1, out) {}

    virtual TENSOR::Tensor forward(const TENSOR::Tensor& input) = 0;
    virtual ~Layer() = default;
};

class Linear : public Layer {
public:
    Linear(const size_t in, const size_t out) : Layer(in, out) {}
    TENSOR::Tensor forward(const TENSOR::Tensor& input) override;
};

} // namespace LAYER