#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"
#include <algorithm>
#include <random>
#include <memory>
#include <vector>
#include <iostream>

int main() {
    const size_t BATCH      = 4;
    const size_t INPUT_DIM  = 8;
    const size_t OUTPUT_DIM = 4;
    const int    EPOCHS     = 100000;
    const double LR         = 0.001;

    try {
        // ── 1. Synthetic data ────────────────────────────────────────────
        // Single fixed batch: the network should overfit to it,
        // so loss must decrease monotonically — a clean sanity check.

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        TENSOR::Tensor input(BATCH, INPUT_DIM);
        {
            std::vector<double> d(BATCH * INPUT_DIM);
            std::generate(d.begin(), d.end(), [&]() { return dist(rng); });
            input.setData(d);
        }

        TENSOR::Tensor target(BATCH, OUTPUT_DIM);
        {
            std::vector<double> d(BATCH * OUTPUT_DIM);
            std::generate(d.begin(), d.end(), [&]() { return dist(rng); });
            target.setData(d);
        }

        // ── 2. Network topology: 8 → 16 → 16 → 4 ────────────────────────
        // Weights are Xavier-initialised inside the Layer constructor.

        CONT::Sequential network;
        network.add(std::make_shared<LAYER::Linear>(INPUT_DIM, 16));
        network.add(std::make_shared<LAYER::Linear>(16,        16));
        network.add(std::make_shared<LAYER::Linear>(16,        OUTPUT_DIM));

        // ── 3. Loss ──────────────────────────────────────────────────────
        LFUN::MSE loss;

        // ── 4. Training loop ─────────────────────────────────────────────
        for (int epoch = 0; epoch <= EPOCHS; ++epoch) {

            // Forward pass
            TENSOR::Tensor output = network.forward(input);

            // Scalar loss for monitoring
            TENSOR::Tensor residual_buf(BATCH, OUTPUT_DIM);
            double L = loss.residuals(output, target, residual_buf);

            // Gradient dL/dY — shape (BATCH, OUTPUT_DIM)
            TENSOR::Tensor grad = loss.gradient(output, target);

            // Backward pass (reverse through layers)
            network.backward(grad);

            // SGD parameter update
            network.update(LR);

            if (epoch % 100 == 0)
                std::cout << "Epoch " << epoch << "  loss: " << L << "\n";
        }

        std::cout << "\nTraining complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}