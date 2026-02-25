#ifndef GLL_H
#define GLL_H

#include <iostream>

namespace gll {
class Basis {
public:
  /**
   * @brief Constructor of Lagrange basis
   * @param p Order of the basis
   * @param q Number of points (overintegration)
   */
  Basis(const int p, const int q);
  const double *getQuads() const { return quads; }
  const double *getWeights() const { return weights; }
  const int getOrder() const { return p; }
  const int getQ() const { return q; }
  const double *getD() const { return D; }

  ~Basis();
  friend std::ostream &operator<<(std::ostream &, const Basis &);

private:
  const int p;
  const int q;
  double *D;
  double *quads;
  double *weights;
};
}; // namespace gll

#endif
