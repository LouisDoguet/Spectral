#include <cmath>
#include "gll.h"

/**
 * @breif Computes Bonnet recursion formula to approximate Legendre poly at order P
 * @param P Order of the LP
 * @param xi double valiu [-1,1]
 * @return double
 */
inline double Bonnet(int P, double xi){
    if (P==0) return 1.;
    if (P==1) return xi;
    double L_nm2 = 1.;
    double L_nm1 = xi;
    double L_n = 0.;
    for (int n=2; n<P+1; n++){
	L_n = ((2.*n - 1.) * xi * L_nm1 - (n-1.) * L_nm2) / n;
	L_nm2 = L_nm1;
	L_nm1 = L_n;
    }
    return L_n;
}


/**
 * @brief Computes the derivative of the LP of order P at point xo
 * @param P Order of the LP
 * @param xi Value in [-1,1]
 * @return double
 */
inline double Lpp(int P, double xi){
    return (P / (1-xi*xi)) * (Bonnet(P-1,xi) - xi*Bonnet(P,xi));
}


/**
 * @brief Computes the double derivative of the LP of order P at point xo
 * @param P Order of the LP
 * @param xi Value in [-1,1]
 * @return double
 */
inline double Lppp(int P, double xi){
    return -(P*(P+1)/(1-xi*xi))*Bonnet(P,xi);
}


/**
 * @brief Computes the Derivative matrix D
 * @param D Matrix [P+1, P+1] storing the derivative coeffs
 * @param quads Positions of the quadrature points
 * @param p Order of the LP
 * @return void
 */
void computeDerivative(double* D, double* quads, const int p){

    int N = p;
    int nNodes = p + 1;

    for (int i = 0; i < nNodes; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < nNodes; ++j) {
            if (i != j) {
                // Off-diagonal formula using Legendre values at the nodes
                double Li = Bonnet(N, quads[i]);
                double Lj = Bonnet(N, quads[j]);
                
                D[i * nNodes + j] = (Li / Lj) * (1.0 / (quads[i] - quads[j]));
                rowSum += D[i * nNodes + j];
            }
        }
        D[i * nNodes + i] = -rowSum;
    }
}


/**
 * @brief Computes the quadrature points of the LP of order P on [-1,1]
 * @param quads Quadrature points array
 * @param P Order of the LP
 * @return void
 */
void setQuads(double* quads, const int P){
    
    quads[0] = -1.0;
    quads[P] = 1.0;

    for (int k=1; k<=(P+1)/2; ++k){
	double xi = -cos(M_PI*k / P);
	double eps;
	double temp;
	for (int iter=0; iter<50; ++iter){
	    temp = xi - (Lpp(P,xi)/Lppp(P,xi));
	    eps = fabs(temp - xi);
	    xi = temp;
	    if (eps < 1e-15) break;
	}
	quads[k] = xi;
	quads[P-k] = -xi;
    }
}


/**
 * @brief Computes the weights of the quadratures of the LP of order P on [-1,1]
 * @param weights Weights of the quadratures
 * @param quad Quadratures of the LP
 * @param P Order of the LP
 * @return void
 */
void setWeights(double* weights, const double* quads, const int P){
    for (int i=0; i<P+1; ++i){
	double Lpi = Bonnet(P, quads[i]);
	weights[i] = 2 / (P*(P+1)* (Lpi*Lpi));
    }
}


namespace gll{
    Basis::Basis(const int p) : p(p) {
	quads = new double[p+1];
	weights = new double[p+1];
	D = new double[(p+1)*(p+1)];
	setQuads(quads, p);
	setWeights(weights, quads, p);
	computeDerivative(D, quads, p);
    }


    Basis::~Basis() {
	delete[] quads;
	delete[] weights;
    }    
    
    std::ostream &operator<<(std::ostream &os, const Basis& b) {
	os << "----- BASIS -----" << std::endl
	   << "ORDER  : " << b.p << std::endl;
	return os;
    }
}

