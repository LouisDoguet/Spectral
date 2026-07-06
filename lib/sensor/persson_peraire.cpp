#include "sensor.h"
#include "../space/element.h"
#include "../math/math.h"
#include <cmath>

static const double PI = 3.14159265358979323846;

double sens::PerssonPeraire::SmoothnessIndicator(elm::Element& elem, int truncation){
    const int N = elem.getBasis()->getOrder()+1;
    const int id_trunc = N - truncation;
    const double* modes = elem.getModes();

    double S  = mat::inner_product(modes, modes, N);
    if (S < 1e-14) return 0.0;

    // High-mode energy: only the top `truncation` modes
    double Se = mat::inner_product(modes + id_trunc, modes + id_trunc, truncation);

    return Se / S;
}

double* sens::PerssonPeraire::getSensor(elm::Element& elem, int truncation){
    const int N = elem.getBasis()->getOrder() + 1;
    double* S = new double[N];
    std::fill(S, S+N, this->SmoothnessIndicator(elem, truncation));
    return S;
}

double* sens::PerssonPeraire::getViscosity(elm::Element& elem, int truncation, double s0, double kappa, double eps0){
    elem.computeLegendreCoefficients();

    double Se = this->SmoothnessIndicator(elem, truncation);
    const int N = elem.getBasis()->getOrder()+1;
    double* res = new double[N]();   // zero-initialised

    if (Se < 1e-14) return res;      // perfectly smooth: no viscosity

    double logSe = std::log10(Se);

    if (logSe >= s0 - kappa && logSe <= s0 + kappa) {
        double eps = eps0 / 2.0 * (1.0 + std::sin(PI * (logSe - s0) / (2.0 * kappa)));
        for (int i = 0; i < N; ++i) res[i] = eps;
    } else if (logSe > s0 + kappa) {
        for (int i = 0; i < N; ++i) res[i] = eps0;
    }

    return res;
}
