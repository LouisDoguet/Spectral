#ifndef HYBRID_ALPHA_H
#define HYBRID_ALPHA_H

#include "../../neural/network.h"
#include "../space/mesh.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace solver {

/**
 * @brief Normalized Legendre modal-energy spectrum of an element's density.
 *
 * Input feature for the hybrid-DGSEM alpha policy. Same spectral-decay signal
 * the Persson-Peraire indicator uses:
 *   feat[j] = m_j^2 / sum_k m_k^2,   j = 0..P
 * A smooth element keeps energy in low modes; a discontinuous element leaks
 * into high modes. A flat element maps to [1,0,...,0] so alpha ~ 0 by
 * construction.
 */
inline void modal_energy_features(elm::Element *E, int n,
                                  std::vector<double> &out) {
  E->computeLegendreCoefficients();
  const double *mc = E->getModes();
  out.assign(n, 0.0);
  double total = 0.0;
  for (int j = 0; j < n; ++j)
    total += mc[j] * mc[j];
  if (total < 1e-30) {
    out[0] = 1.0;
    return;
  }
  for (int j = 0; j < n; ++j)
    out[j] = mc[j] * mc[j] / total;
}

/**
 * @brief Trained policy network that predicts the hybrid-DGSEM blending factor.
 *
 * Replaces the Persson-Peraire modal indicator inside HybridDGSEM. Input is
 * the normalized modal energy spectrum of a single element (P+1 values);
 * output is a scalar alpha in [0,1] (Sigmoid). Because the policy is
 * element-local it is independent of the number of elements and can be reused
 * across mesh sizes for a fixed polynomial order P.
 */
class HybridAlphaNet {
public:
  explicit HybridAlphaNet(const std::string &model_path);

  /**
   * @brief Fill per-element raw alpha predictions in [0,1].
   * @param mesh  Mesh whose current state is queried.
   * @param alpha Output vector (resized to number of elements).
   */
  void fillAlpha(mesh::Mesh *mesh, std::vector<double> &alpha);

private:
  std::string model_path_;
  std::unique_ptr<Network> network_;
};

} // namespace solver

#endif // HYBRID_ALPHA_H
