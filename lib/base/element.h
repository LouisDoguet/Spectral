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
	    Element(const int id, gll::Basis* sharedBasis) : id(id), basis(sharedBasis){};

	private:
	    const int id;
	    gll::Basis* basis;

	friend std::ostream &operator<<(std::ostream&, const Element&);	    

    };
}

#endif
