#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"
#include <iostream>

class Network {
public:
    Network(std::shared_ptr<CONT::Sequential> container, std::shared_ptr<LFUN::LossFunction> loss_function, double learning_rate) : 
        container(container),
        loss_function(loss_function),
        l(learning_rate)    
    {};

    void train(const TENSOR::Tensor& input, const TENSOR::Tensor& target, int epochs){
        for (int epoch = 0; epoch <= epochs; ++epoch) {

            // Forward pass
            TENSOR::Tensor output = container->forward(input);

            // Scalar loss for monitoring

            TENSOR::Tensor residual_buf(output.n_rows, output.n_cols);
            double L = loss_function->residuals(output, target, residual_buf);

            // Gradient dL/dY — shape (BATCH, OUTPUT_DIM)
            TENSOR::Tensor grad = loss_function->gradient(output, target);

            // Backward pass (reverse through layers)
            container->backward(grad);

            // SGD parameter update
            container->update(l);

            if (epoch % 100 == 0)
                std::cout << "Epoch " << epoch << "  loss: " << L << "\n";
        }
    };

    TENSOR::Tensor predict (const TENSOR::Tensor& input);

private:
    std::shared_ptr<CONT::Sequential> container;
    std::shared_ptr<LFUN::LossFunction> loss_function;
    double l;

};

#endif