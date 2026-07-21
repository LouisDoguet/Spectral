#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "../space/mesh.h"
#include "../sensor/sensor.h"
#include <string>

namespace diff {

class _Diffusion {
public:
  _Diffusion(const std::string name) : name(name) {};
  virtual void apply(mesh::Mesh *mesh);

  std::string getName() { return name; }

private:
  const std::string name;
};

class Constant : public _Diffusion {
public:
  Constant(double eps) : _Diffusion("CONSTANT"), epsilon(eps) {};
  void apply(mesh::Mesh *mesh) override;

private:
  double epsilon;
};

class Custom : public _Diffusion {
public:
  Custom(double *eps_array)
      : _Diffusion("CUSTOM"), eps_array(eps_array) {};
  void apply(mesh::Mesh *mesh) override;

private:
  double *eps_array;
};

class PerssonPeraire : public _Diffusion {
public:
  PerssonPeraire(int truncation, double s0, double kappa, double eps0)
      : _Diffusion("PERSSON_PERAIRE"), truncation(truncation), s0(s0), kappa(kappa), eps0(eps0) {}
  void apply(mesh::Mesh *mesh) override;

private:
  int truncation;
  double s0, kappa, eps0;
  sens::PerssonPeraire sensor;
};

} // namespace DIFF

#endif
