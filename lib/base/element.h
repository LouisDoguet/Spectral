#ifndef ELEMENT_H
#define ELEMENT_H

#include "gll.h"

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
	    Element(const int p);
	    ~Element();

	    double* getQuads(){ return quads; }
	    double* getWeights(){ return weights; }

	private:
	    const int p;
	    double* quads;
	    double* weights;

	friend std::ostream &operator<<(std::ostream&, const Element&);	    

    };
}

#endif
