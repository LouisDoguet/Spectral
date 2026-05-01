#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <string>


namespace ACTI {
class Activation {
public:
    Activation(std::string name) : name(name) {};
    virtual void apply(TENSOR::Tensor& tensor) = 0;
protected:
    std::string name;
};

class ReLU : public Activation {
public:
    ReLU() : Activation("ReLU") {}
    void apply(TENSOR::Tensor& tensor) override;
};

class SoftMax : public Activation {
public:
    SoftMax() : Activation("SoftMax") {};
    void apply(TENSOR::Tensor& tensor) override;
};

}

#endif