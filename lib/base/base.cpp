#include "base.h"
#include <cmath>

namespace base {
_Basis::_Basis(std::string name, const int p) : name(name), p(p) {
  quads = new double[p + 1];
  weights = new double[p + 1];
  D = new double[(p + 1) * (p + 1)];
}

_Basis::~_Basis() {
  delete[] quads;
  delete[] weights;
  delete[] D;
}

std::ostream &operator<<(std::ostream &os, const _Basis &b) {
  os << "----- BASIS -----" << std::endl 
     << "NAME   : " << b.name << std::endl
     << "ORDER  : " << b.p << std::endl;
  return os;
}
} // namespace base
