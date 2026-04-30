#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <cmath>
#include <string>
#include "tensor.h"

namespace LFUN {
class LossFunction {
public:
    LossFunction(std::string name) : name(name) {};
    virtual double residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) = 0;
    virtual TENSOR::Tensor gradient(TENSOR::Tensor val, TENSOR::Tensor ref) = 0;
    virtual ~LossFunction() = default;  // also add this
protected:
    std::string name;
};

class MSE : public LossFunction {
public:
    MSE() : LossFunction("MeanSquaredError") {};
    double residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) override;
    TENSOR::Tensor gradient(TENSOR::Tensor val, TENSOR::Tensor ref) override;
    ~MSE() = default;
};

}

#endif