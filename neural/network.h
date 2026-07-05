/**
 * @file network.h
 * @brief Full network header
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

/// @brief Class for a Neural Network object
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

    /// @brief Predict the output from a given input
    /// @param input Tensor of input
    /// @return Tensor of output
    TENSOR::Tensor predict(const TENSOR::Tensor& input) {
        return container->forward(input);
    }


    /// @brief Save the neural newtork
    /// @param path Path of the `.nn` file
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

        static constexpr uint32_t MAGIC   = 0x544E4E53; // "SNNT"
        static constexpr uint32_t VERSION = 1;
        f.write(reinterpret_cast<const char*>(&MAGIC),   4);
        f.write(reinterpret_cast<const char*>(&VERSION), 4);
        f.write(reinterpret_cast<const char*>(&l),       8);

        uint32_t n = (uint32_t)container->layers.size();
        f.write(reinterpret_cast<const char*>(&n), 4);

        for (const auto& layer : container->layers)
            layer->serialize(f);

        if (!f) throw std::runtime_error("Write error while saving model");
    }

    /// @brief Load an existing neural network
    /// @param path Path of the `.nn` file
    /// @param loss Loss function
    /// @return `Network` object
    static Network load(const std::string& path, std::shared_ptr<LFUN::LossFunction> loss) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open model file: " + path);

        uint32_t magic, version;
        double lr;
        f.read(reinterpret_cast<char*>(&magic),   4);
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&lr),      8);

        if (magic   != 0x544E4E53) throw std::runtime_error("Invalid model file (bad magic)");
        if (version != 1)          throw std::runtime_error("Unsupported model version");

        uint32_t n_layers;
        f.read(reinterpret_cast<char*>(&n_layers), 4);

        auto seq = std::make_shared<CONT::Sequential>();
        for (uint32_t i = 0; i < n_layers; ++i) {
            uint8_t  act_id;
            uint64_t in, out;
            f.read(reinterpret_cast<char*>(&act_id), 1);
            f.read(reinterpret_cast<char*>(&in),     8);
            f.read(reinterpret_cast<char*>(&out),    8);

            auto layer = makeLayer(act_id, in, out);
            layer->loadWeightsBias(f);
            seq->add(layer);
        }

        if (!f) throw std::runtime_error("Read error while loading model");
        return Network(seq, loss, lr);
    }

private:
    static std::shared_ptr<LAYER::_Layer> makeLayer(uint8_t act_id, uint64_t in, uint64_t out) {
        switch (act_id) {
            case 0: return std::make_shared<LAYER::ReLU>(in, out);
            case 1: return std::make_shared<LAYER::Sigmoid>(in, out);
            default: throw std::runtime_error("Unsupported activation id: " + std::to_string(act_id));
        }
    }

    std::shared_ptr<CONT::Sequential> container;
    std::shared_ptr<LFUN::LossFunction> loss_function;
    double l;

};

#endif