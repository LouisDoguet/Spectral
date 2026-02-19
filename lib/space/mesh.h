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
	     * @param basis Basis of the elements
	     * @param xL Beginning of the mesh
	     * @param xR End of the mesh
	     */
	    Mesh(const int n, gll::Basis* basis, double xL, double xR);
	    /**
	     * @brief Construct a new mesh, with initial parameters
	     * @param n Mesh number of points
	     * @param basis Basis of the elements
	     * @param xL Beginning of the mesh
	     * @param xR End of the mesh
	     * @param init_u1 U1 initial values (assigned to the entire element)
	     * @param init_u2 U2 initial values (assigned to the entire element)
	     * @param init_u3 U3 initial values (assigned to the entire element)
		 * @param u1_L Left BC
		 * @param u2_L 
		 * @param u3_L 
		 * @param u1_R Right BC
		 * @param u2_R 
		 * @param u3_R 
	     */
	    Mesh(const int n, gll::Basis* basis, double xL, double xR, 
			 double* init_u1, double* init_u2, double* init_u3, 
			 double u1_L, double u2_L, double u3_L, double u1_R, double u2_R, double u3_R);
	   
	    /// GETTERS 
	    const elem::Element* getElem(int i) const { return elem[i]; }
	    elem::Element* getElem(int i) { return elem[i]; }
	    int getNumElements() const { return n; }
	    // Getters for global contiguous buffers
	    double* getGlobalU1() { return global_rho; }
	    double* getGlobalU2() { return global_rhou; }
	    double* getGlobalU3() { return global_e; }
	    int getTotalPoints() const { return n * (elem[0]->getBasis()->getOrder() + 1); }
	    
	    /// Flux solving routines
	    void computeElements();	// Computes df/dx
	    void computeInterfaces();	// Computes the Reimann problem at interface
	    /**
	     * @brief Find R(U) for an element (dU/dt = -dF/dx)
	     * From the initialized U, for all elements:
	     * - Computes F
	     * - Applies the divergence (dFdx)
	     * - Applies the Reimann correction at the elements boundaries
	     * For the first and last elements:
	     * - Applies Boundary Conditions
	     */
	    void computeResidual();

	    /// Boundary conditions
	    void applyDirichlet();


	    ~Mesh();

        private:
	    const int n;
	    elem::Element** elem;

	    /// Unified variables
	    double* global_rho;
	    double* global_rhou;
	    double* global_e;

		/// Boudary conditions
		double u1_L;
		double u2_L;
		double u3_L;
		double u1_R;
		double u2_R;
		double u3_R;
	    
	friend std::ostream &operator<<(std::ostream&, const Mesh&);	    
    };
}

#endif
