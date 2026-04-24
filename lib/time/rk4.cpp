#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cblas.h>
#include "rk4.h"
#include "../math/math.h"
#include "../base/gll.h"

namespace solver {

    RK4::RK4(mesh::Mesh* mesh, int n_plot) : m(mesh) {
        total_points = m->getTotalPoints();
        int P = m->getElem(0)->getBasis()->getOrder();
        this->n_plot = (n_plot > 0) ? n_plot : (P + 1);

        rho_n = new double[total_points];
        rhou_n = new double[total_points];
        e_n = new double[total_points];

        rho_acc = new double[total_points];
        rhou_acc = new double[total_points];
        e_acc = new double[total_points];

        global_df1 = new double[total_points];
        global_df2 = new double[total_points];
        global_df3 = new double[total_points];
    }

    void RK4::save_state() {
        cblas_dcopy(total_points, m->getGlobalU1(), 1, rho_n, 1);
        cblas_dcopy(total_points, m->getGlobalU2(), 1, rhou_n, 1);
        cblas_dcopy(total_points, m->getGlobalU3(), 1, e_n, 1);
        std::memset(rho_acc, 0, total_points * sizeof(double));
        std::memset(rhou_acc, 0, total_points * sizeof(double));
        std::memset(e_acc, 0, total_points * sizeof(double));
    }

    void RK4::collect_residuals() {
        int n_elem = m->getNumElements();
        int n_quads = total_points / n_elem;
        for (int e = 0; e < n_elem; ++e) {
            const elem::Element* el = m->getElem(e);
            std::memcpy(&global_df1[e*n_quads], el->getDivF1(), n_quads * sizeof(double));
            std::memcpy(&global_df2[e*n_quads], el->getDivF2(), n_quads * sizeof(double));
            std::memcpy(&global_df3[e*n_quads], el->getDivF3(), n_quads * sizeof(double));
        }
    }

    void RK4::set_stage_state(double dt, double coeff) {
        double alpha = -dt * coeff;
        cblas_dcopy(total_points, rho_n, 1, m->getGlobalU1(), 1);
        cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
        cblas_dcopy(total_points, e_n, 1, m->getGlobalU3(), 1);
        collect_residuals();
        cblas_daxpy(total_points, alpha, global_df1, 1, m->getGlobalU1(), 1);
        cblas_daxpy(total_points, alpha, global_df2, 1, m->getGlobalU2(), 1);
        cblas_daxpy(total_points, alpha, global_df3, 1, m->getGlobalU3(), 1);
    }

    void RK4::accumulate_stage(double coeff) {
        collect_residuals();
        cblas_daxpy(total_points, -coeff, global_df1, 1, rho_acc, 1);
        cblas_daxpy(total_points, -coeff, global_df2, 1, rhou_acc, 1);
        cblas_daxpy(total_points, -coeff, global_df3, 1, e_acc, 1);
    }

    void RK4::finalize_step(double dt) {
        double alpha = dt / 6.0;
        cblas_dcopy(total_points, rho_n, 1, m->getGlobalU1(), 1);
        cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
        cblas_dcopy(total_points, e_n, 1, m->getGlobalU3(), 1);
        cblas_daxpy(total_points, alpha, rho_acc, 1, m->getGlobalU1(), 1);
        cblas_daxpy(total_points, alpha, rhou_acc, 1, m->getGlobalU2(), 1);
        cblas_daxpy(total_points, alpha, e_acc, 1, m->getGlobalU3(), 1);
    }

    void RK4::step(double dt) {
        save_state();
        m->computeResidual();
        accumulate_stage(1.0);
        set_stage_state(dt, 0.5); 
        m->computeResidual();
        accumulate_stage(2.0);
        set_stage_state(dt, 0.5); 
        m->computeResidual();
        accumulate_stage(2.0);
        set_stage_state(dt, 1.0); 
        m->computeResidual();
        accumulate_stage(1.0);
        finalize_step(dt);
    }

    void RK4::run(double T_final, double dt, int save_freq, std::string prefix) {
        int n_steps = std::ceil(T_final / dt);
        std::cout << "--- Starting Simulation ---" << std::endl;
        for (int step = 0; step <= n_steps; ++step) {
            if (step % save_freq == 0) {
		std::printf("Timestep : %5d/%5d \n", step, n_steps);
		export_results(step, step * dt, prefix);
            }
            this->step(dt);
        }
        write_pvd(prefix);
        std::cout << "--- Simulation Finished ---" << std::endl;
	int nelem = this->m->getNumElements();
	int nquad = this->m->getElem(0)->getBasis()->getOrder()+1;
	std::cout << "Mesh size : " << nelem << std::endl;
	std::cout << "Quads     : " << nquad << std::endl;
	std::cout << "Nodes     : " << nelem*nquad << std::endl;
	std::cout << "Timesteps : " << n_steps << std::endl;
	std::cout << "--- TOTAL OPER : " << n_steps*nelem*nquad << std::endl;
    }

    void RK4::export_results(int step, double time, std::string prefix) {
        std::stringstream ss;
        ss << prefix << "_" << std::setfill('0') << std::setw(6) << step << ".vtu";
        std::string full_path = ss.str();
        std::ofstream file(full_path);

        std::string basename = full_path;
        size_t last_slash = full_path.find_last_of("/\\");
        if (last_slash != std::string::npos)
            basename = full_path.substr(last_slash + 1);
        exported_files.push_back({time, basename});

        int n_elem = m->getNumElements();
        const gll::Basis* basis = m->getElem(0)->getBasis();
        int P = basis->getOrder();
        const double* quads   = basis->getQuads();
        const double* weights = basis->getWeights();

        int n_nodes = n_elem * n_plot;
        int n_cells = n_elem * (n_plot - 1);

        // Uniform reference points in [-1, 1]
        double* ref_pts = new double[n_plot];
        for (int i = 0; i < n_plot; ++i)
            ref_pts[i] = -1.0 + 2.0 * i / (n_plot - 1);

        double* c1 = new double[P + 1];
        double* c2 = new double[P + 1];
        double* c3 = new double[P + 1];

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <UnstructuredGrid>\n";
        file << "    <Piece NumberOfPoints=\"" << n_nodes << "\" NumberOfCells=\"" << n_cells << "\">\n";

        file << "      <Points>\n";
        file << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int e = 0; e < n_elem; ++e) {
            double xL = m->getElem(e)->getX(0);
            double dx = m->getElem(e)->getX(P) - xL;
            for (int i = 0; i < n_plot; ++i)
                file << xL + (ref_pts[i] + 1.0) / 2.0 * dx << " 0.0 0.0 ";
        }
        file << "\n        </DataArray>\n      </Points>\n";

        file << "      <Cells>\n";
        file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int e = 0; e < n_elem; ++e) {
            int offset = e * n_plot;
            for (int i = 0; i < n_plot - 1; ++i) file << offset + i << " " << offset + i + 1 << " ";
        }
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        int cur = 0;
        for (int i = 0; i < n_cells; ++i) { cur += 2; file << cur << " "; }
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < n_cells; ++i) file << "3 ";
        file << "\n        </DataArray>\n      </Cells>\n";

        file << "      <PointData>\n";
        file << "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n";
        for (int e = 0; e < n_elem; ++e) {
            mat::computeLegendreCoeffs(c1, m->getElem(e)->getU1(), quads, weights, P);
            for (int i = 0; i < n_plot; ++i)
                file << mat::evalLegendreExpansion(ref_pts[i], c1, P) << " ";
        }
        file << "\n        </DataArray>\n";

        file << "        <DataArray type=\"Float64\" Name=\"velocity\" format=\"ascii\">\n";
        for (int e = 0; e < n_elem; ++e) {
            mat::computeLegendreCoeffs(c1, m->getElem(e)->getU1(), quads, weights, P);
            mat::computeLegendreCoeffs(c2, m->getElem(e)->getU2(), quads, weights, P);
            for (int i = 0; i < n_plot; ++i) {
                double rho_val  = mat::evalLegendreExpansion(ref_pts[i], c1, P);
                double rhou_val = mat::evalLegendreExpansion(ref_pts[i], c2, P);
                file << rhou_val / rho_val << " ";
            }
        }
        file << "\n        </DataArray>\n";

        file << "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n";
        const double gamma = 1.4;
        for (int e = 0; e < n_elem; ++e) {
            mat::computeLegendreCoeffs(c1, m->getElem(e)->getU1(), quads, weights, P);
            mat::computeLegendreCoeffs(c2, m->getElem(e)->getU2(), quads, weights, P);
            mat::computeLegendreCoeffs(c3, m->getElem(e)->getU3(), quads, weights, P);
            for (int i = 0; i < n_plot; ++i) {
                double rho_val  = mat::evalLegendreExpansion(ref_pts[i], c1, P);
                double rhou_val = mat::evalLegendreExpansion(ref_pts[i], c2, P);
                double e_val    = mat::evalLegendreExpansion(ref_pts[i], c3, P);
                double p = (gamma - 1.0) * (e_val - 0.5 * rhou_val * rhou_val / rho_val);
                file << p << " ";
            }
        }
        file << "\n        </DataArray>\n";

        file << "      </PointData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
        file.close();

        delete[] ref_pts;
        delete[] c1; delete[] c2; delete[] c3;
    }

    void RK4::write_pvd(std::string prefix) {
        std::ofstream file(prefix + ".pvd");
        file << "<?xml version=\"1.0\"?>" << std::endl;
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
        file << "  <Collection>" << std::endl;
        for (auto& entry : exported_files) {
            file << "    <DataSet timestep=\"" << entry.first << "\" group=\"\" part=\"0\" file=\"" << entry.second << "\"/>" << std::endl;
        }
        file << "  </Collection>" << std::endl;
        file << "</VTKFile>" << std::endl;
        file.close();
        std::cout << "VTK Database written to " << prefix << ".pvd" << std::endl;
    }

    RK4::~RK4() {
        delete[] rho_n; delete[] rhou_n; delete[] e_n;
        delete[] rho_acc; delete[] rhou_acc; delete[] e_acc;
        delete[] global_df1; delete[] global_df2; delete[] global_df3;
    }

}
