#ifndef MESH_H
#define MESH_H
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
	    Mesh(const size_t n);
	    ~Mesh();

        private:
	    const size_t n;
	    double* rho;
	    double* rhou;
	    double* e;
    
	friend std::ostream &operator<<(std::ostream&, const Mesh&);	    
    };
}

#endif
