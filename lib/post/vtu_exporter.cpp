#include "vtu_exporter.h"
#include "../base/base.h"
#include "../math/math.h"
#include "../space/element.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace post {

VTUExporter::VTUExporter(mesh::Mesh* mesh, int n_plot) : m(mesh), n_plot(n_plot) {
    addField(fieldRho());
    addField(fieldVelocity());
    addField(fieldPressure());
    addField(fieldAViscosity());
    addField(fieldBasisIndicator());
    // lap_pressure and div_laplacian are opt-in: register them explicitly with
    // addField(VTUExporter::fieldLapPressure()) or addSensorField(...) when needed.
}

void VTUExporter::addField(ScalarField field) {
    fields.push_back(std::move(field));
}

void VTUExporter::addSensorField(const std::string& name, sens::_Sensor* sensor) {
    addField({name,
              [sensor](elem::Element& elem,
                       const double* ref_pts, int n_plot,
                       const double* quads, const double* weights, int P,
                       double* out) {
                  double* s = sensor->getSensor(elem);
                  if (!s) { std::fill(out, out + n_plot, 0.0); return; }
                  double* c = new double[P + 1];
                  mat::computeLegendreCoeffs(c, s, quads, weights, P);
                  for (int i = 0; i < n_plot; ++i)
                      out[i] = mat::evalLegendreExpansion(ref_pts[i], c, P);
                  delete[] c;
                  delete[] s;
              }});
}

void VTUExporter::addElemField(const std::string& name,
                               std::function<double(elem::Element&)> fn) {
    addField({name,
              [fn](elem::Element& elem,
                   const double* /*ref_pts*/, int n_plot,
                   const double* /*quads*/, const double* /*weights*/, int /*P*/,
                   double* out) {
                  double val = fn(elem);
                  for (int i = 0; i < n_plot; ++i) out[i] = val;
              }});
}

void VTUExporter::write(int step, double time, const std::string& prefix) {
    std::stringstream ss;
    ss << prefix << "_" << std::setfill('0') << std::setw(6) << step << ".vtu";
    std::string full_path = ss.str();
    std::ofstream file(full_path);

    std::string basename = full_path;
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos)
        basename = full_path.substr(last_slash + 1);
    exported_files.push_back({time, basename});

    const int n_elem = m->getNumElements();
    // The mesh's primary basis only fixes P and the geometry; per-element
    // quadrature data is queried inside the field-fill loop so adaptive meshes
    // (different elements on different bases) export correctly.
    const int P = m->getElem(0)->getBasis()->getOrder();

    // In nodal mode every element is sampled at its own P+1 quadrature nodes;
    // otherwise at a uniform grid of n_plot points.
    const int np = nodal_mode ? (P + 1) : n_plot;

    const int n_nodes = n_elem * np;
    const int n_cells = n_elem * (np - 1);

    // Uniform reference points (used only when not in nodal mode).
    double* ref_pts = new double[np];
    for (int i = 0; i < np; ++i)
        ref_pts[i] = -1.0 + 2.0 * i / (np - 1);

    // ── Header ───────────────────────────────────────────────────────────
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
            "byte_order=\"LittleEndian\">\n"
         << "  <UnstructuredGrid>\n"
         << "    <Piece NumberOfPoints=\"" << n_nodes
         << "\" NumberOfCells=\"" << n_cells << "\">\n";

    // ── Points ───────────────────────────────────────────────────────────
    file << "      <Points>\n"
         << "        <DataArray type=\"Float64\" Name=\"Points\" "
            "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int e = 0; e < n_elem; ++e) {
        if (nodal_mode) {
            // True physical positions of the computed (quadrature) nodes.
            for (int i = 0; i < np; ++i)
                file << m->getElem(e)->getX(i) << " 0.0 0.0 ";
        } else {
            double xL = m->getElem(e)->getX(0);
            double dx = m->getElem(e)->getX(P) - xL;
            for (int i = 0; i < np; ++i)
                file << xL + (ref_pts[i] + 1.0) / 2.0 * dx << " 0.0 0.0 ";
        }
    }
    file << "\n        </DataArray>\n      </Points>\n";

    // ── Cells ────────────────────────────────────────────────────────────
    file << "      <Cells>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int e = 0; e < n_elem; ++e) {
        int offset = e * np;
        for (int i = 0; i < np - 1; ++i)
            file << offset + i << " " << offset + i + 1 << " ";
    }
    file << "\n        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    int cur = 0;
    for (int i = 0; i < n_cells; ++i) { cur += 2; file << cur << " "; }
    file << "\n        </DataArray>\n"
         << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < n_cells; ++i) file << "3 ";
    file << "\n        </DataArray>\n      </Cells>\n";

    // ── PointData ────────────────────────────────────────────────────────
    double* buf = new double[np];
    file << "      <PointData>\n";
    for (const auto& field : fields) {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" format=\"ascii\">\n";
        for (int e = 0; e < n_elem; ++e) {
            const base::_Basis* eb = m->getElem(e)->getBasis();
            // In nodal mode, sampling at the element's own quadrature nodes
            // makes the interpolation exact -> raw computed nodal values.
            const double* rp = nodal_mode ? eb->getQuads() : ref_pts;
            field.fill(*m->getElem(e), rp, np,
                       eb->getQuads(), eb->getWeights(), eb->getOrder(), buf);
            for (int i = 0; i < np; ++i) file << buf[i] << " ";
        }
        file << "\n        </DataArray>\n";
    }
    file << "      </PointData>\n";

    file << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
    file.close();

    delete[] ref_pts;
    delete[] buf;
}

void VTUExporter::writePVD(const std::string& prefix) {
    std::ofstream file(prefix + ".pvd");
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"Collection\" version=\"0.1\" "
            "byte_order=\"LittleEndian\">\n"
         << "  <Collection>\n";
    for (const auto& entry : exported_files)
        file << "    <DataSet timestep=\"" << entry.first
             << "\" group=\"\" part=\"0\" file=\"" << entry.second << "\"/>\n";
    file << "  </Collection>\n</VTKFile>\n";
    file.close();
}

// ── Built-in field factories ──────────────────────────────────────────────────

ScalarField VTUExporter::fieldRho() {
    return {"rho",
            [](elem::Element& elem,
               const double* ref_pts, int n_plot,
               const double* /*quads*/, const double* /*weights*/, int /*P*/,
               double* out) {
                elem.getBasis()->interpolate(elem.getU1(), ref_pts, n_plot, out);
            }};
}

ScalarField VTUExporter::fieldVelocity() {
    return {"velocity",
            [](elem::Element& elem,
               const double* ref_pts, int n_plot,
               const double* /*quads*/, const double* /*weights*/, int P,
               double* out) {
                const int N = P + 1;
                const double* rho  = elem.getU1();
                const double* rhou = elem.getU2();
                double* u_nodes = new double[N];
                for (int i = 0; i < N; ++i) u_nodes[i] = rhou[i] / rho[i];
                elem.getBasis()->interpolate(u_nodes, ref_pts, n_plot, out);
                delete[] u_nodes;
            }};
}

ScalarField VTUExporter::fieldPressure() {
    return {"pressure",
            [](elem::Element& elem,
               const double* ref_pts, int n_plot,
               const double* /*quads*/, const double* /*weights*/, int P,
               double* out) {
                const double gm1 = 0.4;
                const int N = P + 1;
                const double* rho  = elem.getU1();
                const double* rhou = elem.getU2();
                const double* e    = elem.getU3();
                double* p_nodes = new double[N];
                for (int i = 0; i < N; ++i)
                    p_nodes[i] = gm1 * (e[i] - 0.5 * rhou[i] * rhou[i] / rho[i]);
                elem.getBasis()->interpolate(p_nodes, ref_pts, n_plot, out);
                delete[] p_nodes;
            }};
}

ScalarField VTUExporter::fieldAViscosity() {
    return {"a_viscosity",
            [](elem::Element& elem,
               const double* ref_pts, int n_plot,
               const double* /*quads*/, const double* /*weights*/, int /*P*/,
               double* out) {
                elem.getBasis()->interpolate(elem.getAV(), ref_pts, n_plot, out);
            }};
}

ScalarField VTUExporter::fieldBasisIndicator() {
    mesh::Mesh* mesh_ptr = m;
    return {"basis_id",
            [mesh_ptr](elem::Element& elem,
                       const double* /*ref_pts*/, int n_plot,
                       const double* /*quads*/, const double* /*weights*/, int /*P*/,
                       double* out) {
                const base::_Basis* eb  = elem.getBasis();
                const base::_Basis* alt = mesh_ptr->getAltBasis();
                const double val = (alt != nullptr && eb == alt) ? 1.0 : 0.0;
                for (int i = 0; i < n_plot; ++i) out[i] = val;
            }};
}

ScalarField VTUExporter::fieldLapPressure() {
    return {"lap_pressure",
            [](elem::Element& elem,
               const double* ref_pts, int n_plot,
               const double* /*quads*/, const double* /*weights*/, int /*P*/,
               double* out) {
                const double* lap = elem.computePressureLaplacian();
                elem.getBasis()->interpolate(lap, ref_pts, n_plot, out);
                delete[] const_cast<double*>(lap);
            }};
}

} // namespace post
