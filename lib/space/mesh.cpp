#include <iostream>
#include <cmath>
#include <iomanip>

#include "mesh.h"
#include "../phy/physics.h"

namespace mesh {

    Mesh::Mesh(const int n, gll::Basis* basis, double xL, double xR) : n(n) {
	double dx_mesh = xR - xL;
	double dx = (double) dx_mesh / (n);
	
        int nquads = basis->getOrder() + 1;
        global_rho = new double[n * nquads];
        global_rhou = new double[n * nquads];
        global_e = new double[n * nquads];

	this->elem = new elem::Element*[n]; 
	
	double x_iter = xL;
	for (int e = 0; e<n; ++e) {
	    elem[e] = new elem::Element(e, basis, x_iter, x_iter + dx, 
                                        &global_rho[e*nquads], &global_rhou[e*nquads], &global_e[e*nquads]);
	    elem[e]->setFlux(); 
	    x_iter += dx;
	}
    }	    
    
    Mesh::Mesh(const int n, gll::Basis* basis, double xL, double xR, double* init_u1, double* init_u2, double* init_u3) : n(n) {
	double dx_mesh = xR - xL;
	double dx = (double) dx_mesh / (n);

	/// Global buffer	
        int nquads = basis->getOrder() + 1;
        global_rho = new double[n * nquads];
        global_rhou = new double[n * nquads];
        global_e = new double[n * nquads];

	/// Pointers to the elements
	this->elem = new elem::Element*[n]; 
	
	double x_iter = xL;
	for (int e = 0; e<n; ++e) {
	    /// Assigns initial conditions to the fglobal buffer
            for (int q=0; q<nquads; q++) {
                global_rho[e*nquads + q] = init_u1[e];
                global_rhou[e*nquads + q] = init_u2[e];
                global_e[e*nquads + q] = init_u3[e];
            }

	    /// Construct the element e with for values it's position in the global buffer
	    elem[e] = new elem::Element(e, basis, x_iter, x_iter + dx, 
                                        &global_rho[e*nquads], &global_rhou[e*nquads], &global_e[e*nquads]);
	    /// Computes F from U
	    elem[e]->setFlux();
	    x_iter += dx;
	}
    }


    void Mesh::computeElements(){
	for (int e=0; e<n; ++e){
	    elem[e]->computeDivFlux();
	}
    }


    void Mesh::computeInterfaces() {
        const int P = elem[0]->getBasis()->getOrder();
        const double* w = elem[0]->getBasis()->getWeights();
    
        for (int e = 0; e < n - 1; ++e) {
            /// Select iteracting elements
	    elem::Element* LeftElem = elem[e];
            elem::Element* RightElem = elem[e + 1];
	    
	    /// Set up of the useful values
            double u1L = *(LeftElem->getU1(P)); double u1R = *(RightElem->getU1(0));
            double u2L = *(LeftElem->getU2(P)); double u2R = *(RightElem->getU2(0));
            double u3L = *(LeftElem->getU3(P)); double u3R = *(RightElem->getU3(0));
    
            double f1L = *(LeftElem->getF1(P)); double f1R = *(RightElem->getF1(0));
            double f2L = *(LeftElem->getF2(P)); double f2R = *(RightElem->getF2(0));
            double f3L = *(LeftElem->getF3(P)); double f3R = *(RightElem->getF3(0));
    
	    /// Compute pressure
            double pL, pR;
            phy::getP(&pL, &u1L, &u2L, &u3L, 1);
            phy::getP(&pR, &u1R, &u2R, &u3R, 1);
            
	    /// Compute max wave speed
            double lambdaL = reimann::computeMaxWaveSpeed(u1L, u2L/u1L, pL);
            double lambdaR = reimann::computeMaxWaveSpeed(u1R, u2R/u1R, pR);
            double lambda  = std::max(lambdaL, lambdaR);
    
	    /// Reimann problem
            double f1star = reimann::Rusanov(f1L, f1R, u1L, u1R, lambda);
            double f2star = reimann::Rusanov(f2L, f2R, u2L, u2R, lambda);
            double f3star = reimann::Rusanov(f3L, f3R, u3L, u3R, lambda);
	    
	    /// Apply weight and jacobian
	    double invWJ_L = (1.0 / w[P]) * *(LeftElem->getInvJ());
	    double invWJ_R = (1.0 / w[0]) * *(RightElem->getInvJ());

	    /// Add the correction to the elements
	    LeftElem->correctDivF1(P, invWJ_L * (f1star - f1L));
	    LeftElem->correctDivF2(P, invWJ_L * (f2star - f2L));
	    LeftElem->correctDivF3(P, invWJ_L * (f3star - f3L));

	    RightElem->correctDivF1(0, invWJ_R * (f1R - f1star));
	    RightElem->correctDivF2(0, invWJ_R * (f2R - f2star));
	    RightElem->correctDivF3(0, invWJ_R * (f3R - f3star));        
	}
    }

    void Mesh::applyDirichlet() {
        const int P = elem[0]->getBasis()->getOrder();
        const double* w = elem[0]->getBasis()->getWeights();
    
        // Background External State
        double u1_ext = 1.0; double u2_ext = 1.0; double u3_ext = 2.5; 
        double f1_ext = 1.0; double f2_ext = 2.0; double f3_ext = 3.5; 
    
        // --- LEFT BOUNDARY ---
        double u1_int_L = *(elem[0]->getU1(0));
        double u2_int_L = *(elem[0]->getU2(0));
        double u3_int_L = *(elem[0]->getU3(0));
        double f1_int_L = *(elem[0]->getF1(0));
        double f2_int_L = *(elem[0]->getF2(0));
        double f3_int_L = *(elem[0]->getF3(0));
        
        // Dynamic lambda with safety factor for stability
        double p_int_L;
        phy::getP(&p_int_L, &u1_int_L, &u2_int_L, &u3_int_L, 1);
        double lam_int_L = reimann::computeMaxWaveSpeed(u1_int_L, u2_int_L/u1_int_L, p_int_L);
        double lambda_L = std::max(2.2, lam_int_L) * 1.1; 
    
        double f1s_L = reimann::Rusanov(f1_ext, f1_int_L, u1_ext, u1_int_L, lambda_L);
        double f2s_L = reimann::Rusanov(f2_ext, f2_int_L, u2_ext, u2_int_L, lambda_L);
        double f3s_L = reimann::Rusanov(f3_ext, f3_int_L, u3_ext, u3_int_L, lambda_L);
    
        double invWJ_L = (1.0 / w[0]) * *(elem[0]->getInvJ());
        elem[0]->correctDivF1(0, invWJ_L * (f1_int_L - f1s_L));
        elem[0]->correctDivF2(0, invWJ_L * (f2_int_L - f2s_L));
        elem[0]->correctDivF3(0, invWJ_L * (f3_int_L - f3s_L));
    
        // --- RIGHT BOUNDARY ---
        int last = n - 1;
        double u1_int_R = *(elem[last]->getU1(P));
        double u2_int_R = *(elem[last]->getU2(P));
        double u3_int_R = *(elem[last]->getU3(P));
        double f1_int_R = *(elem[last]->getF1(P));
        double f2_int_R = *(elem[last]->getF2(P));
        double f3_int_R = *(elem[last]->getF3(P));
    
        double p_int_R;
        phy::getP(&p_int_R, &u1_int_R, &u2_int_R, &u3_int_R, 1);
        double lam_int_R = reimann::computeMaxWaveSpeed(u1_int_R, u2_int_R/u1_int_R, p_int_R);
        double lambda_R = std::max(2.2, lam_int_R) * 1.1;
    
        double f1s_R = reimann::Rusanov(f1_int_R, f1_ext, u1_int_R, u1_ext, lambda_R);
        double f2s_R = reimann::Rusanov(f2_int_R, f2_ext, u2_int_R, u2_ext, lambda_R);
        double f3s_R = reimann::Rusanov(f3_int_R, f3_ext, u3_int_R, u3_ext, lambda_R);
    
        double invWJ_R = (1.0 / w[P]) * *(elem[last]->getInvJ());
        elem[last]->correctDivF1(P, invWJ_R * (f1s_R - f1_int_R));
        elem[last]->correctDivF2(P, invWJ_R * (f2s_R - f2_int_R));
        elem[last]->correctDivF3(P, invWJ_R * (f3s_R - f3_int_R));
    }

    void Mesh::computeResidual(){
	for (int e=0; e<n; ++e){
	    elem[e]->setFlux();
	}
	this->computeElements();
	this->computeInterfaces();
	this->applyDirichlet();
    }

    Mesh::~Mesh() {
        for (int e=0; e<n; e++) delete elem[e];
	delete[] elem;
        delete[] global_rho;
        delete[] global_rhou;
        delete[] global_e;
    }

    std::ostream &operator<<(std::ostream &os, const Mesh& m) {
    	os << "----- MESH -----" << std::endl
	   << "N. ELEM  : " << m.n << std::endl;
	return os;
    };

}
