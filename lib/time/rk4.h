#ifndef RK4_H
#define RK4_H

#include <vector>
#include <string>
#include "../space/mesh.h"

namespace solver {

    /**
    * @brief Optimized Runge-Kutta 4th order solver using contiguous buffers and BLAS
    */
    class RK4 {
        public:
            RK4(mesh::Mesh* mesh);
            void step(double dt);
            
            /**
             * @brief Run the simulation loop
             * @param T_final Total simulation time
             * @param dt Time step
             * @param save_freq Frequency of data export (every N steps)
             * @param prefix Output file prefix
             */
            void run(double T_final, double dt, int save_freq, std::string prefix);
    
            /**
             * @brief Export current state to CSV
             */
            void export_results(int step, double time, std::string prefix);
            
	    ~RK4();
            
        private:
            mesh::Mesh* m;
            int total_points;
                    
            // Tracking for VTK PVD Collection
            std::vector<std::pair<double, std::string>> exported_files;
            void write_pvd(std::string prefix);
            
	    // Contiguous buffers to store state at t_n
	    double* rho_n;
	    double* rhou_n;
	    double* e_n;

	    // Contiguous buffers to accumulate stage updates
	    double* rho_acc;
	    double* rhou_acc;
	    double* e_acc;

	    // Temporary buffers for global divF (used for BLAS vector ops)
	    double* global_df1;
	    double* global_df2;
	    double* global_df3;

	    void save_state();
	    void set_stage_state(double dt, double coeff);
	    void accumulate_stage(double coeff);
	    void finalize_step(double dt);
	    void collect_residuals();
    };

}

#endif
