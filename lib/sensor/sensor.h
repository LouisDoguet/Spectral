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
};

class PerssonPeraire : public _Sensor {
private:
    double SmoothnessIndicator(elem::Element& elem, int truncation);

public:
    PerssonPeraire() : _Sensor("PerssonPeraire") {};
    /**
     * @brief Apply the sensor to the element, returning the dissipation value
     * @param elem `Element` to apply the sensor to
     * @return None
     */
    double* getViscosity(elem::Element& elem, int truncation, double s0, double kappa, double eps0);
};

} // namespace sens


#endif