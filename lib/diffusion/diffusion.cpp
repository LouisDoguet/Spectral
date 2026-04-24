#include <algorithm>
#include "diffusion.h"

namespace DIFF {

void Diffusion::apply(mesh::Mesh *mesh) {}

void Constant::apply(mesh::Mesh *mesh) {
  int n = mesh->getElem(0)->getBasis()->getOrder() + 1;
  double *eps_nodes = new double[n];
  std::fill(eps_nodes, eps_nodes + n, epsilon);
  for (int i = 0; i < mesh->getNumElements(); ++i)
    mesh->getElem(i)->applyViscosity(eps_nodes);
  delete[] eps_nodes;
}

} // namespace DIFF
