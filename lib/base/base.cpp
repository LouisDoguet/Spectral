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
  os << "----- BASIS -----" << std::endl 
     << "NAME   : " << b.name << std::endl
     << "ORDER  : " << b.p << std::endl;
  return os;
}
} // namespace base
