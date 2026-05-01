#include "layer.h"

TENSOR::Tensor LAYER::_Layer::forward(const TENSOR::Tensor& input) {
    this->input_cache = input;
    this->preactivation_cache = (input * this->weights).add_bias(this->bias);
    TENSOR::Tensor output = preactivation_cache;
    activation_function->apply(output);
    this->postactivation_cache = output;
    return output;
}

TENSOR::Tensor LAYER::_Layer::backward(const TENSOR::Tensor& grad_output) {
    this->grad_weights = this->input_cache.T() * grad_output;
    this->grad_bias = grad_output.sum_rows();
    TENSOR::Tensor grad_input = grad_output * this->weights.T();
    return grad_input;
}

TENSOR::Tensor LAYER::ReLU::backward(const TENSOR::Tensor& grad_output) {
    // Apply ReLU mask to grad_output using the real preactivation cache
    TENSOR::Tensor masked = grad_output;
    const size_t N = preactivation_cache.n_rows * preactivation_cache.n_cols;
    for (size_t i = 0; i < N; ++i)
        if (preactivation_cache.getData()[i] <= 0.0)
            masked.setData(i, 0.0);

    // Then delegate to the linear backward with the masked gradient
    return _Layer::backward(masked);
}

TENSOR::Tensor LAYER::SoftMax::backward(const TENSOR::Tensor& y_onehot) {
    TENSOR::Tensor grad = postactivation_cache - y_onehot;
    return _Layer::backward(grad);
}

void LAYER::_Layer::update(double learning_rate){
    this->weights = this->weights - (this->grad_weights*learning_rate);
    this->bias = this->bias - (this->grad_bias*learning_rate);
};