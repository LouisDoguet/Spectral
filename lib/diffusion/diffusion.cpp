#include "diffusion.h"
#include "../base/gll.h"
#include "../space/element.h"
#include "../space/mesh.h"
#include <algorithm>
#include <cblas.h>

/**
 * @brief Diffusion term in the element `elem`.
 * @param elem Pointer to an element object
 * @param nodes_eps Diffusion value for each nodes of the element
 * @param n Size of the element (number of nodes)
 * @return None
 */
void diffuse(elem::Element *elem, const double *nodes_eps, const int n) {

  const double *D = elem->getBasis()->getD();
  const double invJ = *(elem->getInvJ());
  double *du1 = new double[n];
  double *du2 = new double[n];
  double *du3 = new double[n];
  double *tmp1 = new double[n];
  double *tmp2 = new double[n];
  double *tmp3 = new double[n];

  // Physical first derivative: invJ * D * u
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, elem->getU1(), 1,
              0., du1, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, elem->getU2(), 1,
              0., du2, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, elem->getU3(), 1,
              0., du3, 1);
  for (int i = 0; i < n; ++i) {
    du1[i] *= nodes_eps[i];
    du2[i] *= nodes_eps[i];
    du3[i] *= nodes_eps[i];
  }

  // Physical second derivative: invJ * D * (eps * invJ * D * u)
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, du1, 1, 0., tmp1,
              1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, du2, 1, 0., tmp2,
              1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, du3, 1, 0., tmp3,
              1);

  // Subtract from divF: diffusion adds to RHS, which means subtracting from
  // divF
  for (int i = 0; i < n; ++i) {
    elem->correctDivF1(i, -tmp1[i]);
    elem->correctDivF2(i, -tmp2[i]);
    elem->correctDivF3(i, -tmp3[i]);
  }

  delete[] du1;
  delete[] du2;
  delete[] du3;
  delete[] tmp1;
  delete[] tmp2;
  delete[] tmp3;
}

namespace DIFF {

void Diffusion::apply(mesh::Mesh *mesh) {}

void Constant::apply(mesh::Mesh *mesh) {
  const gll::Basis *basis = mesh->getBasis();
  int n = basis->getOrder() + 1;
  const int n_elem = mesh->getNumElements();

  double *eps_array = new double[n];

  for (int q = 0; q < n; ++q)
    eps_array[q] = epsilon;

  for (int i = 0; i < n_elem; ++i) {
    elem::Element *e = mesh->getElem(i);
    diffuse(e, eps_array, n);
  }
}

void Custom::apply(mesh::Mesh *mesh) {

  const gll::Basis *basis = mesh->getBasis();
  int n = basis->getOrder() + 1;
  const int n_elem = mesh->getNumElements();

  for (int i = 0; i < n_elem; ++i) {
    elem::Element *e = mesh->getElem(i);
    diffuse(e, eps_array + i * n, n);
  }
}

} // namespace DIFF
