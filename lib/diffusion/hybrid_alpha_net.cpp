#include "hybrid_alpha_net.h"
#include <algorithm>
#include <iostream>

namespace diff {

HybridAlphaNet::HybridAlphaNet(const std::string &model_path)
    : model_path_(model_path), network_(nullptr) {
  auto loss_fn = std::make_shared<LFUN::MSE>();
  network_ = std::make_unique<Network>(Network::load(model_path, loss_fn));
  std::cout << "Hybrid alpha network loaded from: " << model_path << std::endl;
}

void HybridAlphaNet::fillAlpha(mesh::Mesh *mesh, std::vector<double> &alpha) {
  const int n_elem = mesh->getNumElements();
  const int n = mesh->getBasis()->getOrder() + 1;

  alpha.assign(n_elem, 0.0);

  std::vector<double> feat;
  for (int e = 0; e < n_elem; ++e) {
    elm::Element *E = mesh->getElem(e);
    modal_energy_features(E, n, feat);

    TENSOR::Tensor in(1, n);
    in.setData(feat);
    TENSOR::Tensor out = network_->predict(in);

    double a = out.readData()[0];
    alpha[e] = std::max(0.0, std::min(1.0, a));
  }
}

} // namespace diff
