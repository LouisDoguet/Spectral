#include <iostream>
#include <cmath>
#include <iomanip>

#include "mesh.h"

namespace mesh {

    Mesh::Mesh(const int n, gll::Basis* basis, double xL, double xR) : n(n) {
	double dx_mesh = xR - xL;
	double dx = (double) dx_mesh / (n);
	
	this->elem = new elem::Element*[n]; // Array of pointers
	
	double x_iter = xL;
	for (int e = 0; e<n; ++e) {
	    elem[e] = new elem::Element(e, basis, x_iter, x_iter + dx);
	    elem[e]->setFlux(); 
	    x_iter += dx;
	}


    }

    Mesh::~Mesh() {
	delete[] elem;
    }

    std::ostream &operator<<(std::ostream &os, const Mesh& m) {
    	os << "----- MESH -----" << std::endl
	   << "N. ELEM  : " << m.n << std::endl;
	return os;
    };



}
