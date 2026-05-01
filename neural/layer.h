#pragma once
#include "tensor.h"
#include "memory"
#include <random>
#include <algorithm>
#include <string>
#include "activation.h"

namespace LAYER {

class _Layer {
protected:
    const std::string name;
    const size_t in_features;
    const size_t out_features;
    std::shared_ptr<ACTI::Activation> activation_function;
    TENSOR::Tensor weights;
    TENSOR::Tensor bias;
    TENSOR::Tensor input_cache;
    TENSOR::Tensor grad_weights;
    TENSOR::Tensor grad_bias;
    TENSOR::Tensor preactivation_cache;
    TENSOR::Tensor postactivation_cache;

public:
    _Layer(const size_t in, const size_t out, std::shared_ptr<ACTI::Activation> act) : 
        in_features(in), out_features(out), activation_function(act),
        weights(in, out), bias(1, out),
        grad_weights(in, out), grad_bias(1, out) 
    {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (in + out)));
        std::vector<double> w(in * out);
        std::generate(w.begin(), w.end(), [&]() { return dist(rng); });
        weights.setData(w);
    }

    TENSOR::Tensor forward(const TENSOR::Tensor& input);
    virtual TENSOR::Tensor backward(const TENSOR::Tensor& grad_output);
    void update(double learning_rate);
    virtual ~_Layer() = default;

    const size_t getInputSize() { return in_features; }
    const size_t getOutputSize() { return out_features; }

};

class ReLU : public _Layer {
public:
    ReLU(const size_t in, const size_t out) : _Layer(in, out, std::make_shared<ACTI::ReLU>()) {}
    TENSOR::Tensor backward(const TENSOR::Tensor& grad_output) override;
};

class SoftMax : public _Layer {
public:
    SoftMax(const size_t in, const size_t out) : _Layer(in, out, std::make_shared<ACTI::SoftMax>()) {}
    TENSOR::Tensor backward(const TENSOR::Tensor& grad_output) override;
};

} // namespace LAYER