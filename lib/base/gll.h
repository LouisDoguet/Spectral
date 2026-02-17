#ifndef GLL_H
#define GLL_H

#include <iostream>

namespace gll{
    class Basis{
	public:
	    Basis(const int p);
	    const double* getQuads() const {return quads;}
	    const double* getWeights() const {return weights;}
	    const int getOrder() const {return p;}
	    const double* getD() const {return D;}

	    ~Basis();
	    friend std::ostream &operator<<(std::ostream&, const Basis&);

	private:
	    const int p;
	    double* D;
	    double* quads;
	    double* weights;
    };
};

#endif
