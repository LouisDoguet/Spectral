#include <iostream>
#include <cmath>
#include <iomanip>

#include "element.h"
#include "../math/math.h"

namespace elem {
    std::ostream &operator<<(std::ostream &os, const Element& e) {
	os << "----- ELEM -----" << std::endl
	   << "ID  : " << e.id << std::endl;
	return os;
    };

}
