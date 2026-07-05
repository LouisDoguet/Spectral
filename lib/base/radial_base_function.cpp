/**
 * @file radial_base_function.cpp
 * @brief Methods for RBF base
 */

#include "base.h"
#include "../math/math.h"
#include <cblas.h>
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

}

base::RBF::RBF(const int p, const double eps, std::string RBF_name) : _Basis("RadialBaseFunction", p), fname(RBF_name), eps(eps) {
    const int N = this->p + 1;
    this->radial_matrix = new double[N * N];
    this->activated_radial_matrix = new double[N * N];
    this->inv_activ_radial_matrix = new double[N * N];

    rad::setQuads(this->quads, p);
}

void base::RBF::initialize() {
    this->computeRadialMatrix();
    this->activateRadialMatrix();
    this->invertActivatedRadialMatrix();
    this->computeDerivative();
    this->computeWeights();
    this->computeMassMatrix();
}

void base::RBF::computeMassMatrix() {
    const int N = this->p + 1;

    // K_{kl} = int_{-1}^{1} phi(|xi - x_k|) phi(|xi - x_l|) dxi
    // Over-refined composite trapezoidal rule. Integrand is smooth -> O(h^2)
    // convergence, error ~ (2/Q)^2; Q = 4000 gives ~1e-7 — plenty for use.
    const int Q = 4000;
    const double dxi = 2.0 / Q;
    double* K = new double[N * N]();
    double* phi = new double[N];
    for (int q = 0; q <= Q; ++q) {
        const double xi = -1.0 + q * dxi;
        const double wt = (q == 0 || q == Q) ? 0.5 * dxi : dxi;
        for (int k = 0; k < N; ++k) phi[k] = this->kernel(xi - quads[k]);
        for (int k = 0; k < N; ++k)
            for (int l = 0; l < N; ++l)
                K[k * N + l] += wt * phi[k] * phi[l];
    }
    delete[] phi;

    // M = (Phi^{-1})^T * K * Phi^{-1}
    double* tmp = new double[N * N];
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N,
                1.0, inv_activ_radial_matrix, N,
                     K, N,
                0.0, tmp, N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                1.0, tmp, N,
                     inv_activ_radial_matrix, N,
                0.0, M, N);
    delete[] tmp;
    delete[] K;

    // Minv = M^{-1} via LAPACK
    std::copy(M, M + N * N, Minv);
    int n = N, info = 0;
    int* ipiv = new int[N];
    F77NAME(dgetrf)(&n, &n, Minv, &n, ipiv, &info);
    if (info != 0) {
        std::cerr << "RBF::computeMassMatrix: dgetrf info=" << info << std::endl;
        delete[] ipiv;
        return;
    }
    int lwork = -1;
    double work_size = 0.0;
    F77NAME(dgetri)(&n, Minv, &n, ipiv, &work_size, &lwork, &info);
    lwork = (int)work_size;
    double* work = new double[lwork];
    F77NAME(dgetri)(&n, Minv, &n, ipiv, work, &lwork, &info);
    if (info != 0) {
        std::cerr << "RBF::computeMassMatrix: dgetri info=" << info << std::endl;
    }
    delete[] work;
    delete[] ipiv;
}

void base::RBF::activateRadialMatrix() {
    const int N = this->p+1;
    for (int i=0 ; i<N ; ++i){
        for (int j=0; j<N ; ++j){
            this->activated_radial_matrix[i*N + j] = 
                this->kernel(this->radial_matrix[i*N + j]);
        }
    }
}

void base::RBF::computeWeights() {
    const int N = this->p + 1;
    double* I = new double[N];
    for (int k = 0; k < N; ++k) I[k] = this->kernelIntegral(k);
    // w = Phi^{-1} * I  (Phi symmetric -> row-/column-equivalent)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N,
                1.0, this->inv_activ_radial_matrix, N,
                     I, 1,
                0.0, this->weights, 1);
    delete[] I;
}

base::RBF::~RBF() {
    delete[] radial_matrix;
    delete[] activated_radial_matrix;
    delete[] inv_activ_radial_matrix;
}

void base::RBF::interpolate(const double* u, const double* xi, int n_pts, double* out) const {
    const int N = this->p + 1;
    double* lambda = new double[N];
    // lambda = Phi^{-1} u  (one solve, reused for all xi points)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N,
                1.0, this->inv_activ_radial_matrix, N,
                     u, 1,
                0.0, lambda, 1);
    for (int q = 0; q < n_pts; ++q) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            s += lambda[j] * this->kernel(xi[q] - this->quads[j]);
        }
        out[q] = s;
    }
    delete[] lambda;
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
    : base::RBF(p, eps, "InverseMultiQuad") {
    this->initialize();
}

double base::InverseMultiQuadratic::kernel(double r) const {
    return 1.0 / std::sqrt(1.0 + (this->eps * r) * (this->eps * r));
}

double base::InverseMultiQuadratic::kernelIntegral(int k) const {
    // I_k = int_{-1}^{1} 1/sqrt(1 + (eps*(xi - x_k))^2) dxi
    //     = (1/eps) * [ asinh(eps*(1 - x_k)) + asinh(eps*(1 + x_k)) ]
    const double xk = this->quads[k];
    return (1.0 / this->eps) * (std::asinh(this->eps * (1.0 - xk))
                              + std::asinh(this->eps * (1.0 + xk)));
}

void base::InverseMultiQuadratic::computeDerivative() {
    const int N = this->p+1;

    // Phi' : kernel-derivative matrix evaluated at the collocation nodes
    double* Phi_prime = new double[N * N];
    for (int i=0 ; i<N ; ++i){
        for (int j=0; j<N ; ++j){
            Phi_prime[i*N + j] = mat::derivativeInverseMultiQuad(
                this->eps,
                this->quads[i],
                this->quads[j]);
        }
    }

    // D = Phi' * Phi^{-1}  (nodal-to-nodal derivative operator)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0, Phi_prime, N,
                     this->inv_activ_radial_matrix, N,
                0.0, this->D, N);

    delete[] Phi_prime;
}




base::Gaussian::Gaussian(const int p, const double eps)
    : base::RBF(p, eps, "Gaussian") {
    this->initialize();
}

double base::Gaussian::kernel(double r) const {
    return ( exp( - this->eps * r*r ) );
}

double base::Gaussian::kernelIntegral(int k) const {
    // I_k = int_{-1}^{1} exp(-eps*(xi - x_k)^2) dxi
    //     = (1/2) sqrt(pi/eps) [ erf(sqrt(eps)*(1 - x_k)) + erf(sqrt(eps)*(1 + x_k)) ]
    // (using erf odd: -erf(sqrt(eps)*(-1 - x_k)) = erf(sqrt(eps)*(1 + x_k)))
    const double xk = this->quads[k];
    const double s = std::sqrt(this->eps);
    return 0.5 * std::sqrt(M_PI / this->eps)
           * (std::erf(s * (1.0 - xk)) + std::erf(s * (1.0 + xk)));
}

void base::Gaussian::computeDerivative() {
    const int N = this->p+1;

    // Phi' : kernel-derivative matrix evaluated at the collocation nodes
    double* Phi_prime = new double[N * N];
    for (int i=0 ; i<N ; ++i){
        for (int j=0; j<N ; ++j){
            Phi_prime[i*N + j] = mat::derivativeGaussian(
                this->eps,
                this->quads[i],
                this->quads[j]);
        }
    }

    // D = Phi' * Phi^{-1}  (nodal-to-nodal derivative operator)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0, Phi_prime, N,
                     this->inv_activ_radial_matrix, N,
                0.0, this->D, N);

    delete[] Phi_prime;
}