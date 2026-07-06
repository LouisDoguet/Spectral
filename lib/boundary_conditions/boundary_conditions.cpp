#include "boundary_conditions.h"

namespace bc {

// Out-of-line virtual destructor: anchors the _BoundaryConditions vtable in a
// single translation unit (key-function idiom). The Wall/Reflective overrides
// are trivial enough to stay inline in the header.
_BoundaryConditions::~_BoundaryConditions() = default;

} // namespace bc
