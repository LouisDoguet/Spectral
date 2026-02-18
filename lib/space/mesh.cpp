#include <iostream>
#include <cmath>
#include <iomanip>

#include "mesh.h"
#include "../phy/physics.h"

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
	    elem[e] = new elem::Element(e, basis, x_iter, x_iter + dx, init_u1[e], init_u2[e], init_u3[e]);
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

    void Mesh::computeInterfaces() {
        const int P = elem[0]->getBasis()->getOrder();
        const double* w = elem[0]->getBasis()->getWeights();
    
        for (int e = 0; e < n - 1; ++e) {
            elem::Element* LeftElem = elem[e];
            elem::Element* RightElem = elem[e + 1];
    
            // 1. Get pointers to Conserved Variables at the interface
            // Left side uses last node (P), Right side uses first node (0)
            double u1L = *(LeftElem->getU1(P)); double u1R = *(RightElem->getU1(0));
            double u2L = *(LeftElem->getU2(P)); double u2R = *(RightElem->getU2(0));
            double u3L = *(LeftElem->getU3(P)); double u3R = *(RightElem->getU3(0));
    
            // 2. Get pointers to Internal Fluxes
            double f1L = *(LeftElem->getF1(P)); double f1R = *(RightElem->getF1(0));
            double f2L = *(LeftElem->getF2(P)); double f2R = *(RightElem->getF2(0));
            double f3L = *(LeftElem->getF3(P)); double f3R = *(RightElem->getF3(0));
    
            // 3. Compute Pressures and Max Wave Speed (Lambda)
            double pL, pR;
            phy::getP(&pL, &u1L, &u2L, &u3L, 1); // Assuming p calculation for 1 node
            phy::getP(&pR, &u1R, &u2R, &u3R, 1);
            
            double lambdaL = reimann::computeMaxWaveSpeed(u1L, u2L/u1L, pL);
            double lambdaR = reimann::computeMaxWaveSpeed(u1R, u2R/u1R, pR);
            double lambda  = std::max(lambdaL, lambdaR);
    
            // 4. Compute Rusanov Flux for all 3 components
            double f1star = reimann::Rusanov(f1L, f1R, u1L, u1R, lambda);
            double f2star = reimann::Rusanov(f2L, f2R, u2L, u2R, lambda);
            double f3star = reimann::Rusanov(f3L, f3R, u3L, u3R, lambda);
	    
	    // 5. Apply Surface Correction
	    // Scale by 1.0 / (Weight * Jacobian)
	    double invWJ_L = (1.0 / w[P]) * *(LeftElem->getInvJ());
	    double invWJ_R = (1.0 / w[0]) * *(RightElem->getInvJ());

	    // Update Left Element (Right boundary node P)
	    LeftElem->correctDivF1(P, invWJ_L * (f1star - f1L));
	    LeftElem->correctDivF2(P, invWJ_L * (f2star - f2L));
	    LeftElem->correctDivF3(P, invWJ_L * (f3star - f3L));

	    // Update Right Element (Left boundary node 0)
	    RightElem->correctDivF1(0, invWJ_R * (f1R - f1star));
	    RightElem->correctDivF2(0, invWJ_R * (f2R - f2star));
	    RightElem->correctDivF3(0, invWJ_R * (f3R - f3star));        
	}
    }

    void Mesh::applyDirichlet() {
        const int P = elem[0]->getBasis()->getOrder();
        const double* w = elem[0]->getBasis()->getWeights();
    
        // ... (lines omitted for brevity, but I need to replace the whole function or parts)
    
        // --- LEFT BOUNDARY (Element 0, Node 0) ---
        // Initial High-Pressure State (rho=1.0, u=0.0, P=1.0)
        double u1_ext_L = 1.0; 
        double u2_ext_L = 0.0; 
        double u3_ext_L = 2.5; // E = P/(gamma-1) for gamma=1.4
        
        double f1_ext_L = 0.0; // rho*u
        double f2_ext_L = 1.0; // rho*u^2 + p
        double f3_ext_L = 0.0; // u*(E+p)
    
        // Internal State at Node 0 (Pointers dereferenced)
        double u1_int_L = *(elem[0]->getU1(0));
        double u2_int_L = *(elem[0]->getU2(0));
        double u3_int_L = *(elem[0]->getU3(0));
        double f1_int_L = *(elem[0]->getF1(0));
        double f2_int_L = *(elem[0]->getF2(0));
        double f3_int_L = *(elem[0]->getF3(0));
    
        // Approximate Max Wave Speed for Sod High-Pressure side
        double lambdaL = 1.2; 
    
        // Riemann Fluxes (Ghost is Left, Internal is Right)
        double f1star_L = reimann::Rusanov(f1_ext_L, f1_int_L, u1_ext_L, u1_int_L, lambdaL);
        double f2star_L = reimann::Rusanov(f2_ext_L, f2_int_L, u2_ext_L, u2_int_L, lambdaL);
        double f3star_L = reimann::Rusanov(f3_ext_L, f3_int_L, u3_ext_L, u3_int_L, lambdaL);
    
        // Apply scaling 1/(Weight * J)
        double invWJ_L = (1.0 / w[0]) * *(elem[0]->getInvJ());
    
        // Correction for Left boundary node 0
        elem[0]->correctDivF1(0, invWJ_L * (f1_int_L - f1star_L));
        elem[0]->correctDivF2(0, invWJ_L * (f2_int_L - f2star_L));
        elem[0]->correctDivF3(0, invWJ_L * (f3_int_L - f3star_L));
    
        // --- RIGHT BOUNDARY (Last Element, Node P) ---
        int last = n - 1;
        // Initial Low-Pressure State (rho=0.125, u=0.0, P=0.1)
        double u1_ext_R = 0.125; 
        double u2_ext_R = 0.0; 
        double u3_ext_R = 0.25; 
        
        double f1_ext_R = 0.0; 
        double f2_ext_R = 0.1; 
        double f3_ext_R = 0.0;
    
        // Internal State at Node P
        double u1_int_R = *(elem[last]->getU1(P));
        double u2_int_R = *(elem[last]->getU2(P));
        double u3_int_R = *(elem[last]->getU3(P));
        double f1_int_R = *(elem[last]->getF1(P));
        double f2_int_R = *(elem[last]->getF2(P));
        double f3_int_R = *(elem[last]->getF3(P));
    
        double lambdaR = 1.2; 
    
        // Riemann Fluxes (Internal is Left, Ghost is Right)
        double f1star_R = reimann::Rusanov(f1_int_R, f1_ext_R, u1_int_R, u1_ext_R, lambdaR);
        double f2star_R = reimann::Rusanov(f2_int_R, f2_ext_R, u2_int_R, u2_ext_R, lambdaR);
        double f3star_R = reimann::Rusanov(f3_int_R, f3_ext_R, u3_int_R, u3_ext_R, lambdaR);
    
        double invWJ_R = (1.0 / w[P]) * *(elem[last]->getInvJ());
    
        // Correction for Right boundary node P
        elem[last]->correctDivF1(P, invWJ_R * (f1star_R - f1_int_R));
        elem[last]->correctDivF2(P, invWJ_R * (f2star_R - f2_int_R));
        elem[last]->correctDivF3(P, invWJ_R * (f3star_R - f3_int_R));
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
