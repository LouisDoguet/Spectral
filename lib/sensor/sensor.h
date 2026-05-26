#ifndef SENSOR_H
#define SENSOR_H

#include <string>
#include "../space/element.h"

namespace sens
{
class _Sensor {
private:
    std::string name;

public:
    _Sensor(std::string name): name(name) {};
    std::string getName() { return name; }

    /**
     * @brief Returns per-node sensor values for the element (size P+1).
     * Caller owns the returned array and must delete[] it.
     * Default returns nullptr (sensor does not support direct export).
     */
    virtual double* getSensor(elem::Element& elem) { return nullptr; }
    virtual ~_Sensor() = default;
};

class PerssonPeraire : public _Sensor {
private:
    double SmoothnessIndicator(elem::Element& elem, int truncation);

public:
    /**
     * @brief Persson-Peraire sensor
     * 
     * Sensor based on the decay of the highest modes of of the solution.
     * Highly subject to the parameters (`s0`,`kappa`,`eps0`) (see `.getViscosity` method)
     * 
     * @note Possibility to change the energy of the modes measured via `truncation`
     */
    PerssonPeraire() : _Sensor("PerssonPeraire") {};
    /**
     * @brief Apply the sensor to the element, returning the dissipation value
     * @param elem `Element` to apply the sensor to
     * @return None
     */
    double* getViscosity(elem::Element& elem, int truncation, double s0, double kappa, double eps0);
    double* getSensor(elem::Element& elem, int truncation);
};

class DivLaplacian : public _Sensor {
private:
    double* SmoothnessIndicator(elem::Element& elem);

public:
    DivLaplacian() : _Sensor("DivLaplacian") {};
    double* getSensor(elem::Element& elem) override;
};

} // namespace sens


#endif