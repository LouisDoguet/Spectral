#include <iostream>
#include <cmath>

#include "lib/space/mesh.h"
#include "lib/space/element.h"
#include "lib/math/math.h"

int main() {
    const size_t n = 100;
    const int P = 2;
    gll::Basis* B = new gll::Basis(P);
    
    elem::Element* E = new elem::Element(1, B, -1, 1);

    std::cout << *B;
    std::cout << *E;

    const double* q = B->getQuads();
    const double* w = B->getWeights();
    const double* D = B->getD();

    E->computeDivFlux();

    mat::print(q,P+1);
    mat::print(w,P+1);
    mat::print(D,P+1,P+1);

    return 0;
}
