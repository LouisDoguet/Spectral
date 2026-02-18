#ifndef ELEMENT_H
#define ELEMENT_H

#include "../base/gll.h"

namespace elem{
    /**
     * @brief Class storing an element
     */
    class Element {
	public:
	    Element(const int id, gll::Basis* sharedBasis, double xL, double xR);
	    Element(const int id, gll::Basis* sharedBasis, double xL, double xR, 
		    double* rho, double* rhou, double* e);
	    const gll::Basis* getBasis() const { return basis; };

	    void setBasis(gll::Basis* sharedBasis){this->basis = sharedBasis;}
	    void setJ(double xL, double xR);
	    void setID(int ID){this->id=ID;}
	    void setU1(double* rho){this->rho = rho;}
	    void setU2(double* rhou){this->rhou = rhou;}
	    void setU3(double* e){this->e = e;}
	    void setFlux();
	    void computeDivFlux();

	    const int* getID() const { return &id; }
	    const double* getU1() const { return rho; }
	    const double* getU2() const { return rhou; }
	    const double* getU3() const { return e; }
	    const double* getF1() const { return F1; }
	    const double* getF2() const { return F2; }
	    const double* getF3() const { return F3; }
	    const double* getDivF1() const { return divF1; }
	    const double* getDivF2() const { return divF2; }
	    const double* getDivF3() const { return divF3; }

	    ~Element();
	private:
	    double xL;
	    double xR;
	    
	    int id;
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
