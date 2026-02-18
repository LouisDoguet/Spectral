#ifndef MESH_H
#define MESH_H

#include "element.h"
#include "../base/gll.h"

namespace mesh{
    /**
     * @brief Class storing complete mesh
     */
    class Mesh {
	public:
	    /**
	     * @brief Construct a new mesh
	     * @param n Mesh number of points
	     */
	    Mesh(const int n, gll::Basis* basis, double xL, double xR);
	    
	    const elem::Element* getElem(int i) const { return elem[i]; }
	    
	    ~Mesh();

        private:
	    const int n;
	    elem::Element** elem;
	    
	friend std::ostream &operator<<(std::ostream&, const Mesh&);	    
    };
}

#endif
