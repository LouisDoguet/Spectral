#include <iostream>
#include <cmath>
#include "physics.h"
	

namespace phy {

    void getP(double* p, double* rho, double* rhou, double* e, int n){
	double gamma = 1.4;
	for (int i=0; i<n; ++i){
	    double u = rhou[i]/rho[i];
	    p[i]=(gamma - 1)*(e[i] - 1./2.*rho[i]*u*u);
	}
    }
    
    void computeFlux(double* f1, double* f2, double* f3, double* u1, double* u2, double* u3, double* p, int n){	
	for (int i=0; i<n; ++i){
	    f1[i] = u2[i];
	    f2[i] = u1[i]*u2[i] + p[i];
	    f3[i] = u2[i]/u1[i]*(u3[i] + p[i]);
	}
    }

}
