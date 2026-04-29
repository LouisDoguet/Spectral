#include "tensor.h"
#include "layer.h"
#include <iostream>

int main() {
    // Define topology
    const size_t batch_size = 4;
    const size_t input_nodes = 128;
    const size_t hidden_nodes = 8;

    try {
        // Initialize batched input tensor X
        TENSOR::Tensor input(batch_size, input_nodes);

        // Initialize Layer
        LAYER::Linear layer1(input_nodes, hidden_nodes);

        // Execute forward pass: Y = XW + B
        TENSOR::Tensor output = layer1.forward(input);

        std::cout << "Forward propagation successful.\n";
        std::cout << "Output tensor shape: (" << output.n_rows << ", " << output.n_cols << ")\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}