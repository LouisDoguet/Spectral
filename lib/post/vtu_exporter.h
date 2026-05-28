#ifndef VTU_EXPORTER_H
#define VTU_EXPORTER_H

#include "../space/mesh.h"
#include "../sensor/sensor.h"
#include <functional>
#include <string>
#include <vector>

namespace post {

/**
 * @brief Descriptor for a scalar point-data field written to VTU.
 *
 * `fill` is called once per element. It writes `n_plot` values into `out`.
 * Parameters mirror the element's basis so lambdas can do Legendre
 * interpolation when needed.
 */
struct ScalarField {
    std::string name;
    std::function<void(elem::Element& elem,
                       const double* ref_pts, int n_plot,
                       const double* quads, const double* weights, int P,
                       double* out)> fill;
};

/**
 * @brief Writes VTU/PVD output files for a spectral-element mesh.
 *
 * Default fields (rho, velocity, pressure, a_viscosity) are registered
 * automatically. `lap_pressure` and any sensor outputs (e.g. `div_laplacian`)
 * are opt-in: register them with addField(VTUExporter::fieldLapPressure())
 * or addSensorField(name, sensor) before calling run().
 */
class VTUExporter {
public:
    VTUExporter(mesh::Mesh* mesh, int n_plot);

    /** Register a fully custom scalar field. */
    void addField(ScalarField field);

    /**
     * @brief Register a sensor for export.
     *
     * Calls sensor->getSensor(elem) each timestep to get per-node values
     * (size P+1 at GLL nodes), then Legendre-interpolates them to the
     * output grid so the field is smooth in ParaView.
     */
    void addSensorField(const std::string& name, sens::_Sensor* sensor);

    /**
     * @brief Convenience wrapper for element-constant outputs.
     *
     * `fn(elem)` is evaluated once per element; the value is broadcast to
     * all n_plot output points of that element.
     */
    void addElemField(const std::string& name,
                      std::function<double(elem::Element&)> fn);

    /** Write one VTU snapshot and append it to the PVD index. */
    void write(int step, double time, const std::string& prefix);

    /** Flush the PVD collection file. */
    void writePVD(const std::string& prefix);

    // ── Built-in field factories ──────────────────────────────────────────
    static ScalarField fieldRho();
    static ScalarField fieldVelocity();
    static ScalarField fieldPressure();
    static ScalarField fieldAViscosity();
    static ScalarField fieldLapPressure();

private:
    mesh::Mesh* m;
    int n_plot;
    std::vector<ScalarField> fields;
    std::vector<std::pair<double, std::string>> exported_files;
};

} // namespace post

#endif
