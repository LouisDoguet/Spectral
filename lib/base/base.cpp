#include "base.h"
#include "../math/math.h"
#include <cmath>

namespace base {
_Basis::_Basis(std::string name, const int p) : name(name), p(p) {
  const int N = p + 1;
  quads = new double[N];
  weights = new double[N];
  D = new double[N * N];
  M = new double[N * N]();
  Minv = new double[N * N]();
}

_Basis::~_Basis() {
  delete[] quads;
  delete[] weights;
  delete[] D;
  delete[] M;
  delete[] Minv;
}

std::ostream &operator<<(std::ostream &os, const _Basis &b) {
  const int N = b.p + 1;
  os << "----- BASIS -----" << std::endl
     << "NAME   : " << b.name << std::endl
     << "ORDER  : " << b.p << std::endl
     << "NODES  : " << N << std::endl;

  os << "QUADS  : ";
  mat::print(b.quads, N, os);

  os << "WEIGHTS: ";
  mat::print(b.weights, N, os);

  os << "DERIVATIVE MATRIX D [" << N << "x" << N << "] :";
  mat::print(b.D, N, N, os);

  os << "MASS MATRIX M [" << N << "x" << N << "] :";
  mat::print(b.M, N, N, os);

  os << "INVERSE MASS MATRIX Minv [" << N << "x" << N << "] :";
  mat::print(b.Minv, N, N, os);

  return os;
}
} // namespace base
