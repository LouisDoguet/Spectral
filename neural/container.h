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

};
    
} // namespace CONT


#endif