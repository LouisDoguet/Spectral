#include <iostream>
#include <cmath>
#include <iomanip>

#include "element.h"
#include "../math/math.h"

namespace elem {

    Element::Element(const int P) : p(P) {
	quads = new double[p+1]();
	weights = new double[p+1]();
	GetQuads(quads, P);
	GetWeights(weights, quads, P);
    }

    Element::~Element() {
	delete[] quads;
	delete[] weights;
    }

    std::ostream &operator<<(std::ostream &os, const Element& e) {
	os << "----- ELEM -----" << std::endl
	   << "LP ORDER  : " << e.p << std::endl;
	return os;
    };



}
