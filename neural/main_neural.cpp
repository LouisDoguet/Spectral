#include "tensor.h"
#include "layer.h"
#include "container.h"
#include <memory>
#include <vector>
#include <iostream>

int main() {
    // Define topology

    try {
        // Initialize batched input tensor X
        TENSOR::Tensor input(4, 128);

        // Generate Layers
        LAYER::Linear layer1(128, 8);
        LAYER::Linear layer2(8, 16);
        LAYER::Linear layer3(16, 16);
        LAYER::Linear layer4(16, 4);

        CONT::Sequential network;
    
        network.add( std::make_unique<LAYER::Linear>(layer1) );
        network.add( std::make_unique<LAYER::Linear>(layer2) );
        network.add( std::make_unique<LAYER::Linear>(layer3) );
        network.add( std::make_unique<LAYER::Linear>(layer4) );
        

        // Execute forward pass: Y = XW + B
        TENSOR::Tensor output = network.forward(input);

        std::cout << "Forward propagation successful.\n";
        std::cout << "Output tensor shape: (" << output.n_rows << ", " << output.n_cols << ")\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}