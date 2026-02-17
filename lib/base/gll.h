#ifndef GLL_H
#define GLL_H

#include <iostream>

namespace gll{
    class Basis{
	public:
	    Basis(const int p);
	    double* getQuads() {return quads;}
	    double* getWeights() {return weights;}
	    const int getOrder() {return p;}

	    ~Basis();
	    friend std::ostream &operator<<(std::ostream&, const Basis&);

	private:
	    const int p;
	    double* quads;
	    double* weights;
    };
};

#endif
