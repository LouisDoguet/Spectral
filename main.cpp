#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "lib/space/element.h"
#include "lib/base/gll.h"
#include "lib/space/mesh.h"
#include "lib/math/math.h"

int main() {
    const int P = 4;
    const int N = 10;
    gll::Basis* B = new gll::Basis(P);
    std::cout << *B;
    mesh::Mesh M = mesh::Mesh(N, B, 0., 1.);

    for (int i=0; i<N; ++i){
	std::cout << *M.getElem(i);
	mat::print(M.getElem(i)->getBasis()->getQuads(),P+1);
    }

}
