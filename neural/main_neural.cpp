#include "tensor.h"
#include "layer.h"
#include "container.h"
#include "loss_function.h"
#include "activation.h"
#include "network.h"
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

        std::shared_ptr<CONT::Sequential> architecture = std::make_shared<CONT::Sequential>();
        architecture->add(std::make_shared<LAYER::ReLU>(INPUT_DIM, 16));
        architecture->add(std::make_shared<LAYER::ReLU>(16,        16));
        architecture->add(std::make_shared<LAYER::ReLU>(16,        OUTPUT_DIM));

        std::shared_ptr<LFUN::MSE> loss = std::make_shared<LFUN::MSE>();

        Network nwork(architecture, loss, LR);
        nwork.train(input, target, EPOCHS);
        
        std::cout << "\nTraining complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}