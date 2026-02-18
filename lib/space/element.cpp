#include <iostream>
#include <cmath>
#include <iomanip>
#include <cblas.h>
#include <cstring>

#include "element.h"
#include "../math/math.h"
#include "../phy/physics.h"

/**
 * @brief BLAS of the derivative (dgemv)
 * @param dFdx DivFlux to overwrite
 * @param D Derivative matrix
 * @param F Flux
 * @param invJ 1/J, where J jacobian for the base change
 * @param n Size of the vector
 */
void divF(double* dFdx, const double* D, const double* F, const double invJ, const int n){
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n,n, invJ,D,n, F,1, 0.,dFdx,1);
}


namespace elem {

    /**
     * @brief Sets the Jacobian from the element position
     */
    void Element::setJ(double xL, double xR){
	this->xL = xL;	
	this->xR = xR;	
	double dx = xR - xL;
	this->J = dx / 2.; 
	this->invJ = 1 / J; 
    }

    Element::Element(const int id, gll::Basis* sharedBasis, double xL, double xR) : id(id), basis(sharedBasis){
	this->setJ(xL, xR);
	this->ownsMemory = true;
	int n = basis->getOrder()+1;
	rho = new double[n]; rhou = new double[n]; e = new double[n];
	F1 = new double[n]; F2 = new double[n]; F3 = new double[n];
	divF1 = new double[n]; divF2 = new double[n]; divF3 = new double[n];
    }

    Element::Element(const int id, gll::Basis* sharedBasis, double xL, double xR, double rho_init, double rhou_init, double e_init) 
	: id(id), basis(sharedBasis) {
	int nquads = basis->getOrder()+1;
	this->ownsMemory = true;
	rho = new double[nquads]; rhou = new double[nquads]; e = new double[nquads];
	for (int i=0; i<nquads; i++){
	    rho[i]=rho_init; rhou[i]=rhou_init; e[i]=e_init;
	}
	this->setJ(xL, xR);
	F1 = new double[nquads]; F2 = new double[nquads]; F3 = new double[nquads];
	divF1 = new double[nquads]; divF2 = new double[nquads]; divF3 = new double[nquads];
    }

    Element::Element(const int id, gll::Basis* sharedBasis, double xL, double xR,
                    double* external_rho, double* external_rhou, double* external_e)
        : id(id), basis(sharedBasis), rho(external_rho), rhou(external_rhou), e(external_e) {
        
	this->setJ(xL, xR);
        this->ownsMemory = false;
        
	int n = basis->getOrder() + 1;
        
	F1 = new double[n]; 
	F2 = new double[n]; 
	F3 = new double[n];
        divF1 = new double[n]; 
	divF2 = new double[n]; 
	divF3 = new double[n];
    }

    /**
     * @brief Sets the flux from the Euler system solved
     */
    void Element::setFlux() {
	int n = basis->getOrder() + 1;
	double* p = new double[n];
	phy::getP(p, rho, rhou, e, n);
	phy::computeFlux(F1, F2, F3, rho, rhou, e, p, n);
	delete[] p;
    }

    void Element::computeDivFlux(){
	divF(divF1, basis->getD(), F1, invJ, basis->getOrder()+1);
	divF(divF2, basis->getD(), F2, invJ, basis->getOrder()+1);
	divF(divF3, basis->getD(), F3, invJ, basis->getOrder()+1);
    }

    Element::~Element() {
	if (ownsMemory) {
	    delete[] rho; delete[] rhou; delete[] e;
	}
	delete[] F1; delete[] F2; delete[] F3;
	delete[] divF1; delete[] divF2; delete[] divF3;
    }

    std::ostream &operator<<(std::ostream &os, const Element& e) {
	os << "----- ELEM -----" << std::endl
	   << "ID  : " << e.id << std::endl
	   << "xL  : " << e.xL << std::endl
	   << "xR  : " << e.xR << std::endl;
	return os;
    }
}
