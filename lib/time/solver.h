#ifndef SOLVER_H
#define SOLVER_H

#include <ostream>
#include <string>
#include "../space/mesh.h"
#include "../sensor/sensor.h"
#include "../post/vtu_exporter.h"

namespace diff { class _Diffusion; }

namespace solver {

    class _Solver {
        public:
            _Solver(std::string name, mesh::Mesh* mesh, int n_plot = 0);

            virtual void step(double dt) = 0;

            /**
             * @brief Run the simulation loop
             * @param T_final Total simulation time
             * @param dt Time step
             * @param save_freq Frequency of data export (every N steps)
             * @param prefix Output file prefix
             */
            virtual void run(double T_final, double dt, int save_freq, std::string prefix) = 0;

            /**
             * @brief Export current state as a raw binary snapshot for ML training.
             * Layout: int32 n_elem | int32 P | float64 t | float64[n_total] rho | rhou | e
             */
            void export_snapshot(int step, double time, std::string dir);

            void setDiffusion(diff::_Diffusion* diffus) { diffusion = diffus; }
            void setSensor(sens::_Sensor* sensr) { sensor = sensr; }
            void setSnapshotDir(std::string dir) { snapshot_dir = std::move(dir); }

            /// @brief Enable per-element basis adaptation driven by Persson-Peraire.
            /// At the start of every timestep, elements above `s_shock` switch to
            /// `alt`; elements on `alt` below `s_smooth` switch back. `alt` must
            /// share the basis order P of the mesh's primary basis.
            void enableBasisAdaptation(base::_Basis* alt, sens::_Sensor* ssor,
                                       int truncation, double s_shock, double s_smooth) {
                adapt_alt = alt;
                adapt_pp = ssor;
                adapt_trunc = truncation;
                adapt_s_shock = s_shock;
                adapt_s_smooth = s_smooth;
            }

            /** Access the VTU exporter to register extra fields before run(). */
            post::VTUExporter& getExporter() { return *exporter; }

            /** Register a sensor for export by name — calls getSensor() on every element at each output step. */
            void addSensorField(const std::string& name, sens::_Sensor* sensor) {
                exporter->addSensorField(name, sensor);
            }

            ~_Solver();

            friend std::ostream &operator<<(std::ostream &, const _Solver &);

        protected:
            std::string name;
            int total_points;
            int n_plot;
            std::string snapshot_dir;
            mesh::Mesh* m;
            diff::_Diffusion* diffusion = nullptr;
            sens::_Sensor* sensor = nullptr;

            // Basis-adaptation params (used by step() between timesteps)
            base::_Basis*           adapt_alt      = nullptr;
            sens::_Sensor*          adapt_pp       = nullptr;
            int                     adapt_trunc    = 1;
            double                  adapt_s_shock  = 0.0;
            double                  adapt_s_smooth = 0.0;

            post::VTUExporter* exporter;

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
            void collect_residuals();
    };

    /**
    * @brief Optimized Runge-Kutta 4th order solver using contiguous buffers and BLAS
    */
    class RK4 : public _Solver {
        public:
            RK4(mesh::Mesh* mesh, int n_plot = 0) : _Solver("RungeKutta4", mesh, n_plot) {};

            void step(double dt) override;

            /**
             * @brief Run the simulation loop
             * @param T_final Total simulation time
             * @param dt Time step
             * @param save_freq Frequency of data export (every N steps)
             * @param prefix Output file prefix
             */
            void run(double T_final, double dt, int save_freq, std::string prefix) override;

        private:
            void set_stage_state(double dt, double coeff);
            void accumulate_stage(double coeff);
            void finalize_step(double dt);
    };

}

#endif
