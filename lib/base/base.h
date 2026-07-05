/**
 * @file base.h
 * @brief Header for basis definition
 */

#ifndef BASE_H
#define BASE_H

#include <iostream>
#include <string>

namespace base{

	/// @brief Dummy class for Different basis
	class _Basis {
	public:
		_Basis(std::string name, const int p);
		virtual const double* getQuads() const = 0;
		virtual const double* getWeights() const = 0;
		virtual const int getOrder() const = 0;
		virtual const double* getD() const = 0;
		/// @brief Reference mass matrix M_ij = int_{-1}^{1} l_i(xi) l_j(xi) dxi
		virtual const double* getM() const = 0;
		/// @brief Inverse of the reference mass matrix, precomputed at construction.
		virtual const double* getMinv() const = 0;

		/// @brief Evaluates the basis interpolant of nodal values `u` at `n_pts` reference points `xi[k] in [-1,1]`. Writes results to `out`.
		virtual void interpolate(const double* u, const double* xi, int n_pts, double* out) const = 0;

		virtual ~_Basis();

		friend std::ostream &operator<<(std::ostream&, const _Basis&);

	protected:
		std::string name;
		const int p;
	    double* D;
	    double* M;
	    double* Minv;
	    double* quads;
	    double* weights;
	};

    /// @brief Lagrange (nodal) Basis
	/// @note Lagrange/Legendre (nodal/modal) switch possible using the Vandermonde matrix
    class Lagrange : public _Basis {
	public:
	    /// @brief Constructor of Lagrange basis
	    /// @param p Order of the basis
	    Lagrange(const int p);
	    const double* getQuads() const override {return quads;}
	    const double* getWeights() const override {return weights;}
	    const int getOrder() const override {return p;}
	    const double* getD() const override {return D;}
	    const double* getM() const override {return M;}
	    const double* getMinv() const override {return Minv;}
	    void interpolate(const double* u, const double* xi, int n_pts, double* out) const override;
    };

	/// @brief Radial Basis Function Basis
	class RBF : public _Basis {
	public:
		/// @brief Constructor of the RBF
		/// @param p Number of internal nodes
		/// @param eps Radius
		/// @param RBF_name `string` of the name of the RBF
		RBF(const int p, const double eps, std::string RBF_name);
		~RBF() override;

		const double* getQuads()   const override { return quads;   }
		const double* getWeights() const override { return weights; }
		const int     getOrder()   const override { return p;       }
		const double* getD()       const override { return D;       }
		const double* getM()       const override { return M;       }
		const double* getMinv()    const override { return Minv;    }

		/// @brief Generic RBF interpolation: lambda = Phi^{-1} u, then
		///   s(xi) = sum_j lambda_j * kernel(xi - x_j).
		void interpolate(const double* u, const double* xi, int n_pts, double* out) const override;

	protected:
		/// @brief Kernel function phi evaluated at the (signed) radial position r.
		virtual double kernel(double r) const = 0;

		const std::string fname;
		const double eps;
		double* radial_matrix;
		double* activated_radial_matrix;
		double* inv_activ_radial_matrix;

		/// @brief Runs the kernel-dependent construction steps in the right
		///   order. MUST be called from the derived class's constructor body,
		///   not from RBF's own constructor, so that virtual dispatch reaches
		///   the derived overrides.
		void initialize();

		/// @brief Forms the matrix of signed differences `quad_i - quad_j`
		///   (signed form is what the kernel-derivative formula needs).
		void computeRadialMatrix();
		void invertActivatedRadialMatrix();
		/// @brief w_i = sum_k [Phi^{-1}]_{i,k} * int_{-1}^{1} phi(|xi - x_k|) dxi
		void computeWeights();
		/// @brief M_ij = int_{-1}^{1} l_i^{RBF}(xi) l_j^{RBF}(xi) dxi via over-
		///   refined quadrature; inverse computed with LAPACK.
		void computeMassMatrix();

		/// @brief Applies the kernel element-wise to the signed-difference
		///   matrix. Shared across kernels: the kernel itself is virtual.
		void activateRadialMatrix();
		/// @brief Builds the nodal derivative matrix D = Phi' * Phi^{-1}
		virtual void computeDerivative() = 0;
		/// @brief Closed-form integral of the kernel centred at node k over [-1,1]
		virtual double kernelIntegral(int k) const = 0;
	};

	/// @brief Inverse Multiquadratic Radial Basis Function
	class InverseMultiQuadratic : public RBF {
	public:
		InverseMultiQuadratic(const int p, const double eps);
		void computeDerivative() override;
		double kernelIntegral(int k) const override;

	protected:
		double kernel(double r) const override;
	};

	/// @brief Gaussian Radial Basis Function
	class Gaussian : public RBF {
	public:
		Gaussian(const int p, const double eps);
		void computeDerivative() override;
		double kernelIntegral(int k) const override;

	protected:
		double kernel(double r) const override;

	};

};

#endif
