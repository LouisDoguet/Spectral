#include "base.h"
#include "../math/math.h"
#include <cmath>

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dgetrf)(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
    void F77NAME(dgetri)(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
}

namespace rad {

    /**
     * @brief Computes the quadrature points of RBF base
     * @param quads Quadrature points array
     * @param P Order of the LP
     * @return void
     */
    void setQuads(double *quads, const int P) {
        const int Q = P+1;
        const double dx = 2./P;

        for (int q=0; q<Q; ++q) quads[q] = -1. + q*dx;
    }

    /**
     * @brief Computes the weights of the quadratures of the RBF base
     * @param weights Weights of the quadratures
     * @param quad Quadratures of the LP
     * @param P Order of the LP
     * @return void
     */
    void setWeights(double *weights, const int P) {
        std::fill(weights, weights+(P+1), 1.);
    }

}

base::RBF::RBF(const int p, const double eps, std::string RBF_name) : _Basis("RadialBaseFunction", p), fname(RBF_name), eps(eps) {
    const int N = this->p + 1;
    this->radial_matrix = new double[N];
    this->activated_radial_matrix = new double[N];
    this->inv_activ_radial_matrix = new double[N];

    rad::setQuads(this->quads, p);
    rad::setWeights(this->weights, p);
    this->computeRadialMatrix();
    this->activateRadialMatrix();
    this->computeDerivative();
}

void base::RBF::computeRadialMatrix() {
    const int N = this->p+1;
    for (int i=0 ; i<N ; ++i){
        for (int j=i; j<N ; ++j){
            this->radial_matrix[i*N + j] = this->quads[i] - this->quads[j];
            this->radial_matrix[j*N + i] = this->radial_matrix[i*N + j];
        }
    }
}

void base::RBF::invertActivatedRadialMatrix() {
    const int N = this->p+1;

    // Copy activated_radial_matrix to inv_activ_radial_matrix
    std::copy(this->activated_radial_matrix,
              this->activated_radial_matrix + N*N,
              this->inv_activ_radial_matrix);

    // Prepare LAPACK parameters
    int n = N;
    int lda = N;
    int info = 0;
    int* ipiv = new int[N];

    // Step 1: Compute LU factorization via dgetrf
    F77NAME(dgetrf)(&n, &n, this->inv_activ_radial_matrix, &lda, ipiv, &info);
    if (info != 0) {
        std::cerr << "Error: dgetrf failed with info = " << info << std::endl;
        delete[] ipiv;
        return;
    }

    // Step 2: Compute inverse via dgetri with optimal workspace
    int lwork = -1;
    double work_size = 0.0;
    F77NAME(dgetri)(&n, this->inv_activ_radial_matrix, &lda, ipiv, &work_size, &lwork, &info);

    lwork = (int)work_size;
    double* work = new double[lwork];

    F77NAME(dgetri)(&n, this->inv_activ_radial_matrix, &lda, ipiv, work, &lwork, &info);
    if (info != 0) {
        std::cerr << "Error: dgetri failed with info = " << info << std::endl;
    }

    delete[] ipiv;
    delete[] work;
}

base::InverseMultiQuadratic::InverseMultiQuadratic(const int p, const double eps) 
    : base::RBF(p, eps, "InverseMultiQuad") {}

void base::InverseMultiQuadratic::activateRadialMatrix() {
    const int N = this->p+1;
    for (int i=0 ; i<N ; ++i){
        for (int j=0; j<N ; ++j){
            this->activated_radial_matrix[i*N + j] = 
                1 / sqrt( 1 + pow( this->eps*this->radial_matrix[i*N + j] ,2) );
        }
    }
}

void base::InverseMultiQuadratic::computeDerivative() {
    const int N = this->p+1;
    for (int i=0 ; i<N ; ++i){
        for (int j=0; j<N ; ++j){
            this->D[i*N + j] = mat::derivativeInverseMultiQuad(
                this->eps, 
                this->quads[i], 
                this->quads[j]);
        }
    }
}