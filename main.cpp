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

    double* rho = new double[N];
    double* rhou = new double[N];
    double* e = new double[N];

    for (int i=1; i < N; ++i) {
        rho[i-1] = (double) i;
        rhou[i-1] = (double) 3*i ;
        e[i-1] = (double) 5*i;
    }
    mesh::Mesh* M = new mesh::Mesh(N, B, 0., 1., rho, rhou, e);

    for (int i=0; i<3; ++i){
	std::cout << *M->getElem(i);
	mat::print(M->getElem(i)->getBasis()->getQuads(),P+1);
	mat::print(M->getElem(i)->getU1(),P+1);
	mat::print(M->getElem(i)->getU2(),P+1);
	mat::print(M->getElem(i)->getU3(),P+1);
	mat::print(M->getElem(i)->getF1(),P+1);
	mat::print(M->getElem(i)->getF2(),P+1);
	mat::print(M->getElem(i)->getF3(),P+1);
	mat::print(M->getElem(i)->getDivF1(),P+1);
	mat::print(M->getElem(i)->getDivF2(),P+1);
	mat::print(M->getElem(i)->getDivF3(),P+1);
    }
   
    M->computeElements(); 
    for (int i=0; i<3; ++i){
	std::cout << std::endl;
	mat::print(M->getElem(i)->getDivF1(),P+1);
	mat::print(M->getElem(i)->getDivF2(),P+1);
	mat::print(M->getElem(i)->getDivF3(),P+1);
    }
}
