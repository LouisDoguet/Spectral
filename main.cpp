#include <iostream>
#include <cmath>

#include "lib/space/mesh.h"
#include "lib/base/element.h"
#include "lib/math/math.h"

int main() {
    const size_t n = 100;
    const int P = 10;
    mesh::Mesh* M = new mesh::Mesh(n);
    
    elem::Element* E = new elem::Element(P);

    std::cout << *M;
    std::cout << *E;

    double* q = E->getQuads();
    double* w = E->getWeights();

    mat::print(q,P+1);
    mat::print(w,P+1);

    return 0;
}
