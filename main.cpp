#include <iostream>
#include <cmath>

#include "lib/space/mesh.h"
#include "lib/base/element.h"
#include "lib/math/math.h"

int main() {
    const size_t n = 100;
    const int P = 100;
    gll::Basis* B = new gll::Basis(P);
    
    elem::Element* E = new elem::Element(1, B);

    std::cout << *B;
    std::cout << *E;

    double* q = B->getQuads();
    double* w = B->getWeights();

    mat::print(q,P+1);
    mat::print(w,P+1);

    return 0;
}
