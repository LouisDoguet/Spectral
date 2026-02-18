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
    
    Mesh::Mesh(const int n, gll::Basis* basis, double xL, double xR, double* init_u1, double* init_u2, double* init_u3) : n(n) {
	double dx_mesh = xR - xL;
	double dx = (double) dx_mesh / (n);
	
	this->elem = new elem::Element*[n]; // Array of pointers
	
	double x_iter = xL;
	for (int e = 0; e<n; ++e) {
	    elem[e] = new elem::Element(e, basis, x_iter, x_iter + dx, init_u1, init_u2, init_u3);
	    elem[e]->setFlux();
	    x_iter += dx;
	}
    }


    void Mesh::computeElements(){
	for (int e=0; e<n; ++e){
	    elem[e]->computeDivFlux();
	    std::cout << "ElemID " << *elem[e]->getID() << " divergence flux computed." << std::endl;
	}
    }

    void Mesh::computeInterfaces(){
	const int nquads = elem[0]->getBasis()->getOrder();
	for (int e=0; e<n-1; ++e){
	    elem::Element* LeftElem = elem[e];
	    elem::Element* RightElem = elem[e+1];
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
