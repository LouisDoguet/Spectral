#include "sensor.h"
#include "../space/element.h"
#include "../math/math.h"
#include <cmath>
#include "cblas.h"

double* sens::DivLaplacian::getSensor(elem::Element& elem){
    int N = elem.getBasis()->getOrder() + 1;

    const double* divV = elem.computeVelocityDivergence();
    const double* lapP = elem.computePressureLaplacian();
    double EPS1 = 1.;
    double EPS2 = 1.;

    double* Id = new double[N];
    double* Ip = new double[N];
    double* S = new double[N];

    for (int q=0; q<N; ++q) {
        S[q] = EPS1 * Id[q] * std::max(Ip[q], 1e-3);
    }

    delete[] Id;
    delete[] Ip;

    return S;
}

double* sens::DivLaplacian::SmoothnessIndicator(elem::Element& elem){
    return sens::DivLaplacian::getSensor(elem);
}

