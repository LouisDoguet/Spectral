#include <iostream>
#include <cmath>
#include <iomanip>
#include <cblas.h>

#include "element.h"
#include "../math/math.h"
#include "../phy/physics.h"

void divF(double* dFdx, const double* D, const double* F, const double invJ, const int n){
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n,n, invJ,D,n, F,1, 0.,dFdx,1);
}


namespace elem {

    /**
     * @brief Setter for the linear mapping of the Jacobian
     * @param xL Left boundary of the element in the x space
     * @param xR Right boundary of the element in the x space
     * @return void
     */
    void Element::setJ(double xL, double xR){
	this->xL = xL;	
	this->xR = xR;	
	double dx = xR - xL;
	this->J = dx / 2.; /// Jacobian of the linear mapping for xi
	this->invJ = 1 / J; /// Inversion for computational efficiency
    }



    /**
     * @briefs Construct a new Element
     * @param sharedBasis Basis object, shared by all the elements
     * @param xL Left boundary of the element in the x space
     * @param xR Right boundary of the element in the x space
     */
    Element::Element(const int id, gll::Basis* sharedBasis, double xL, double xR) : id(id), basis(sharedBasis){
	
	this->setJ(xL, xR);

	rho = new double[basis->getOrder()+1];
	rhou = new double[basis->getOrder()+1];
	e = new double[basis->getOrder()+1];

	F1 = new double[basis->getOrder()+1];
	F2 = new double[basis->getOrder()+1];
	F3 = new double[basis->getOrder()+1];
	divF1 = new double[basis->getOrder()+1];
	divF2 = new double[basis->getOrder()+1];
	divF3 = new double[basis->getOrder()+1];
    }

    /**
     * @briefs Construct a new Element with imposed values of DENSITY, MOMENTUM, ENERGY
     * @param sharedBasis Basis object, shared by all the elements
     * @param xL Left boundary of the element in the x space
     * @param xR Right boundary of the element in the x space
     * @rho Density array
     * @rhou Momentum array
     * @e Energy array
     */ 
    Element::Element(const int id, gll::Basis* sharedBasis, double xL, double xR, double rho_init, double rhou_init, double e_init) 
	: id(id), basis(sharedBasis) {

	int nquads = basis->getOrder()+1;

	rho = new double[nquads];
	rhou = new double[nquads];
	e = new double[nquads];

	for (int i=0; i<nquads; i++){
	    rho[i]=rho_init;
	    rhou[i]=rhou_init;
	    e[i]=e_init;
	}

	this->setJ(xL, xR);

	F1 = new double[nquads];
	F2 = new double[nquads];
	F3 = new double[nquads];
	divF1 = new double[nquads];
	divF2 = new double[nquads];
	divF3 = new double[nquads];
    }


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
	delete[] rho;
	delete[] rhou;
	delete[] e;
	delete[] F1;
	delete[] F2;
	delete[] F3;
	delete[] divF1;
	delete[] divF2;
	delete[] divF3;
    }

    std::ostream &operator<<(std::ostream &os, const Element& e) {
	os << "----- ELEM -----" << std::endl
	   << "ID  : " << e.id << std::endl
	   << "xL  : " << e.xL << std::endl
	   << "xR  : " << e.xR << std::endl;
	return os;
    }

}
