#ifndef ELEMENT_H
#define ELEMENT_H

#include "../base/gll.h"

namespace elem{
    /**
     * @brief Class storing an element
     */
    class Element {
	public:
	    /**
	     * @brief Construct a new element
	     * @param p Order of the base function
	     */
	    Element(const int id, gll::Basis* sharedBasis, double xL, double xR);
	    Element(const int id, gll::Basis* sharedBasis, double xL, double xR, 
		    double* rho, double* rhou, double* e);
	    const gll::Basis* getBasis() const { return basis; };

	    void setF();
	    void computeDivFlux();

	    ~Element();
	private:
	    const int id;
	    gll::Basis* basis;
	    double J;
	    double invJ;

	    double* rho; 
	    double* rhou; 
	    double* e;

	    double* F1;
	    double* F2;
	    double* F3;

	    double* divF1;
	    double* divF2;
	    double* divF3;

	friend std::ostream &operator<<(std::ostream&, const Element&);	    

    };
}

#endif
