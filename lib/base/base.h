#ifndef BASE_H
#define BASE_H

#include <iostream>
#include <string>

namespace base{

	class _Basis {
	public:
		_Basis(std::string name, const int p);
		virtual const double* getQuads() const = 0;
		virtual const double* getWeights() const = 0;
		virtual const int getOrder() const = 0;
		virtual const double* getD() const = 0;

		/// @brief Evaluates the basis interpolant of nodal values `u`
		///   at `n_pts` reference points `xi[k] in [-1,1]`. Writes results to `out`.
		virtual void interpolate(const double* u, const double* xi, int n_pts, double* out) const = 0;

		virtual ~_Basis();

		friend std::ostream &operator<<(std::ostream&, const _Basis&);

	protected:
		std::string name;
		const int p;
	    double* D;
	    double* quads;
	    double* weights;
	};

    class Lagrange : public _Basis {
	public:
	    /**
	     * @brief Constructor of Lagrange basis
	     * @param p Order of the basis
	     */
	    Lagrange(const int p);
	    const double* getQuads() const override {return quads;}
	    const double* getWeights() const override {return weights;}
	    const int getOrder() const override {return p;}
	    const double* getD() const override {return D;}
	    void interpolate(const double* u, const double* xi, int n_pts, double* out) const override;
    };

	class RBF : public _Basis {
	public:
		RBF(const int p, const double eps, std::string RBF_name);
		~RBF() override;

		const double* getQuads()   const override { return quads;   }
		const double* getWeights() const override { return weights; }
		const int     getOrder()   const override { return p;       }
		const double* getD()       const override { return D;       }

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

		virtual void activateRadialMatrix() = 0;
		/// @brief Builds the nodal derivative matrix D = Phi' * Phi^{-1}
		virtual void computeDerivative() = 0;
		/// @brief Closed-form integral of the kernel centred at node k over [-1,1]
		virtual double kernelIntegral(int k) const = 0;
	};

	class InverseMultiQuadratic : public RBF {
	public:
		InverseMultiQuadratic(const int p, const double eps);
		void activateRadialMatrix() override;
		void computeDerivative() override;
		double kernelIntegral(int k) const override;

	protected:
		double kernel(double r) const override;
	};

};

#endif
