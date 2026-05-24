#include "container.h"
#include "idxreader.h"
#include "layer.h"
#include "loss_function.h"
#include "network.h"
#include "tensor.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

static int argmax(const TENSOR::Tensor &t) {
  const auto &d = t.readData();
  return (int)(std::max_element(d.begin(), d.end()) - d.begin());
}

int main() {
  const std::string TRAIN_IMG = "neural/dataset_mnist/train-images-idx3-ubyte";
  const std::string TRAIN_LBL = "neural/dataset_mnist/train-labels-idx1-ubyte";
  const std::string TEST_IMG = "neural/dataset_mnist/t10k-images-idx3-ubyte";
  const std::string TEST_LBL = "neural/dataset_mnist/t10k-labels-idx1-ubyte";

  const int N_TRAIN = 500;
  const int IMG_SIZE = 784; // 28x28
  const int N_CLASSES = 10;
  const int EPOCHS = 5000;
  const double LR = 0.05;

  try {
    // Load training data
    std::cout << "Loading " << N_TRAIN << " training samples...\n";
    idxDatabase train_img_db(TRAIN_IMG);
    idxLabelDB train_lbl_db(TRAIN_LBL);

    std::vector<double> input_data(N_TRAIN * IMG_SIZE);
    std::vector<double> target_data(N_TRAIN * N_CLASSES, 0.0);

    for (int i = 0; i < N_TRAIN; ++i) {
      Image img = train_img_db.read_image(i);
      int lbl = train_lbl_db.getLabel(i);
      for (int j = 0; j < IMG_SIZE; ++j)
        input_data[i * IMG_SIZE + j] = static_cast<double>(img.getPixel(j));
      target_data[i * N_CLASSES + lbl] = 1.0;
    }

    TENSOR::Tensor input(N_TRAIN, IMG_SIZE);
    TENSOR::Tensor target(N_TRAIN, N_CLASSES);
    input.setData(input_data);
    target.setData(target_data);

    auto arch = std::make_shared<CONT::Sequential>();
    arch->add(std::make_shared<LAYER::ReLU>(IMG_SIZE, 128));
    arch->add(std::make_shared<LAYER::ReLU>(128, 64));
    arch->add(std::make_shared<LAYER::ReLU>(64, 32));
    arch->add(std::make_shared<LAYER::Sigmoid>(32, N_CLASSES));

    auto loss = std::make_shared<LFUN::CrossEntropy>();
    Network net(arch, loss, LR);

    // Train
    net.train(input, target, EPOCHS);

    // Neu
    net.save("neural/MNIST.nn");

    // Evaluate 3 test samples
    idxDatabase test_img_db(TEST_IMG);
    idxLabelDB test_lbl_db(TEST_LBL);

    std::cout << "\n========== Test Predictions ==========\n";
    for (int i = 0; i < 5; ++i) {
      Image img = test_img_db.read_image(i);
      int label = test_lbl_db.getLabel(i);

      std::vector<double> px(IMG_SIZE);
      for (int j = 0; j < IMG_SIZE; ++j)
        px[j] = static_cast<double>(img.getPixel(j));
      TENSOR::Tensor single(1, IMG_SIZE);
      single.setData(px);

      TENSOR::Tensor output = net.predict(single);
      int predicted = argmax(output);

      std::cout << "\n--- Sample " << i << " ---\n";
      std::cout << img << "\n";
      std::cout << "Label    : " << label << "\n";
      std::cout << "Predicted: " << predicted << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
