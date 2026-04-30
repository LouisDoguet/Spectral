#ifndef CONTAINER_H
#define CONTAINER_H

#include <memory>
#include <vector>
#include "layer.h"
#include "tensor.h"

namespace CONT {
class Sequential {
protected:
    std::vector<std::shared_ptr<LAYER::Layer>> layers;

public:
    void add(std::shared_ptr<LAYER::Layer> layer) {
        layers.push_back(layer);
    }

    TENSOR::Tensor forward(TENSOR::Tensor input) {
        for (auto& layer : layers){
            input = layer->forward(input);
        }
        return input;
    }

    TENSOR::Tensor backward(TENSOR::Tensor grad) {
        for (int i = layers.size() - 1; i >= 0; --i)
            grad = layers[i]->backward(grad);
        return grad;
    }

    void update(double learning_rate) {
        for (auto& layer : layers)
            layer->update(learning_rate);
    }

};
    
} // namespace CONT


#endif