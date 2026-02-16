#include <iostream>
#include <cmath>
#include <iomanip>

#include "mesh.h"

namespace mesh {

    Mesh::Mesh(const size_t n) : n(n) {
	rho = new double[n];
	rhou = new double[n];
	e = new double[n];
    }

    Mesh::~Mesh() {
	delete[] rho;
	delete[] rhou;
	delete[] e;
    }

    std::ostream &operator<<(std::ostream &os, const Mesh& m) {
    	os << "----- MESH -----" << std::endl
	   << "N. ELEM  : " << m.n << std::endl;
	return os;
    };



}
