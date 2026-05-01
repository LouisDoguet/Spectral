#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"

class Network {
public:
    Network(std::shared_ptr<CONT::Sequential> container, std::shared_ptr<LFUN::LossFunction> loss_function, double learning_rate) : 
        container(container),
        loss_function(loss_function),
        l(learning_rate)    
    {};

private:
    std::shared_ptr<CONT::Sequential> container;
    std::shared_ptr<LFUN::LossFunction> loss_function;
    double l;

};

#endif