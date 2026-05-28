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
};

#endif
