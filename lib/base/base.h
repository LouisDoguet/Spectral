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

		~_Basis();

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
    };

	class RBF : public _Basis {
	public:
		RBF(const int p, const double eps, std::string RBF_name);

	protected:
		const std::string fname;
		const double eps;
		double* radial_matrix;
		double* activated_radial_matrix;
		double* inv_activ_radial_matrix;

		/// @brief Forms the matrix of the distances of the quadrature `|| quad_i - quad_j ||`
		void computeRadialMatrix();
		virtual void activateRadialMatrix() {};
		void invertActivatedRadialMatrix();
		/// @brief Computes derivative
		virtual void computeDerivative() {};
	};

	class InverseMultiQuadratic : public RBF {
	public:
		InverseMultiQuadratic(const int p, const double eps);
		/// @brief Applies Inverse Multi Quadratic function to the radial matrix
		void activateRadialMatrix() override;
		void computeDerivative() override;
	};

};

#endif
