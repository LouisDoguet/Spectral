#pragma once
#include "tensor.h"
#include "memory"

namespace LAYER {

class Layer {
protected:
    const size_t in_features;
    const size_t out_features;
    TENSOR::Tensor weights;
    TENSOR::Tensor bias;
    TENSOR::Tensor input_cache;
    TENSOR::Tensor grad_weights;
    TENSOR::Tensor grad_bias;

public:
    Layer(const size_t in, const size_t out) : 
        in_features(in), out_features(out),
        weights(in, out), bias(1, out),
        grad_weights(in, out), grad_bias(1, out) {}

    virtual TENSOR::Tensor forward(const TENSOR::Tensor& input) = 0;
    virtual ~Layer() = default;

    virtual std::unique_ptr<Layer> generateLayerFrom(const size_t next_out_features) const = 0;

    const size_t getInputSize() { return in_features; }
    const size_t getOutputSize() { return out_features; }

};

class Linear : public Layer {
public:
    Linear(const size_t in, const size_t out) : Layer(in, out) {}
    TENSOR::Tensor forward(const TENSOR::Tensor& input) override;
    std::unique_ptr<Layer> generateLayerFrom(const size_t next_out_features) const override;
};

} // namespace LAYER