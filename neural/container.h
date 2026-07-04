#ifndef CONTAINER_H
#define CONTAINER_H

#include <memory>
#include <vector>
#include "layer.h"
#include "tensor.h"

namespace CONT {
/// @brief Sequential object storing a vector of shared pointers to `_Layers` objects
class Sequential {
public:
    std::vector<std::shared_ptr<LAYER::_Layer>> layers;

    /// @brief Add a layer to the container
    /// @param layer Shared pointer to the layer
    void add(std::shared_ptr<LAYER::_Layer> layer) {
        layers.push_back(layer);
    }

    /// @brief Forward propagation of the entire container
    /// @param input Input of the container
    TENSOR::Tensor forward(TENSOR::Tensor input) {
        for (auto& layer : layers){
            input = layer->forward(input);
        }
        return input;
    }

    /// @brief Backward propagation of the entire container
    /// @param grad Gradient of the output of the container
    TENSOR::Tensor backward(TENSOR::Tensor grad) {
        for (int i = layers.size() - 1; i >= 0; --i)
            grad = layers[i]->backward(grad);
        return grad;
    }

    /// @brief Update the weights of the neural network
    /// @param learning_rate 
    void update(double learning_rate) {
        for (auto& layer : layers)
            layer->update(learning_rate);
    }

};
    
} // namespace CONT


#endif