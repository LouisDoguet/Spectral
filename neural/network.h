#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"

class Network {
public:
    Network(CONT::Sequential container, LFUN::LossFunction loss_function, double learning_rate) : 
        container(container),
        loss_function(loss_function),
        l(learning_rate)    
    {};

private:
    CONT::Sequential container;
    LFUN::LossFunction loss_function;
    double l;

};

#endif