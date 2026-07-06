#ifndef HYBRID_ALPHA_NET_H
#define HYBRID_ALPHA_NET_H

#include "../../neural/network.h"
#include "../space/mesh.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace diff {

/**
 * @brief Normalized Legendre modal-energy spectrum of an element's density.
 *
 * This is the input feature for the hybrid-DGSEM alpha policy. It is the same
 * spectral-decay signal the Persson-Peraire indicator is built on:
 *   feat[j] = m_j^2 / sum_k m_k^2,   j = 0..P
 * where m_j are the Legendre coefficients of the nodal density.
 *
 * Why this representation (vs. a standardized nodal profile):
 *   - It is magnitude-aware in the right sense: a *smooth* element keeps almost
 *     all energy in the low modes, while a *discontinuous* element leaks energy
 *     into the high modes. The two are clearly separable.
 *   - A flat/constant element maps to [1,0,...,0] (energy only in mode 0), NOT
 *     to a degenerate zero vector — so constant regions get alpha ~ 0 by
 *     construction instead of a spurious mid-level value.
 *
 * @param E   Element (its density is read; Legendre coefficients are refreshed).
 * @param n   Number of modes (P+1).
 * @param out Output buffer (resized to n).
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
    out[0] = 1.0; // perfectly flat -> all energy in the mean mode
    return;
  }
  for (int j = 0; j < n; ++j)
    out[j] = mc[j] * mc[j] / total;
}

/**
 * @class HybridAlphaNet
 * @brief Trained policy network that predicts the hybrid-DGSEM blending factor.
 *
 * Replaces the Persson-Peraire modal indicator inside HybridDGSEM. The network
 * is *per element*: its input is the standardized density profile of a single
 * element (P+1 values) and its output is a single alpha in [0,1] (Sigmoid).
 * Because the policy is local it is independent of the number of elements and
 * can be reused across mesh sizes for a fixed polynomial order P.
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

} // namespace diff

#endif // HYBRID_ALPHA_NET_H
