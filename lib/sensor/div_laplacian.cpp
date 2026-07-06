#include "sensor.h"
#include "../space/element.h"
#include "../math/math.h"
#include <cmath>
#include "cblas.h"

double* sens::DivLaplacian::getSensor(elm::Element& elem){
    int N = elem.getBasis()->getOrder() + 1;

    const double* divV = elem.computeVelocityDivergence();
    const double* lapP = elem.computePressureLaplacian();
    double EPS1 = 1.;
    double EPS2 = 1.;

    double* S = new double[N];

    for (int q=0; q<N; ++q) {
        S[q] = EPS1 * divV[q] * std::max(lapP[q], 1e-3);
    }

    return S;
}

double* sens::DivLaplacian::SmoothnessIndicator(elm::Element& elem){
    return sens::DivLaplacian::getSensor(elem);
}