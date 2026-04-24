#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "../space/mesh.h"
#include <string>

namespace DIFF {
class Diffusion {
public:
  Diffusion(const std::string name) : name(name) {};
  virtual void apply(mesh::Mesh *mesh);

private:
  const std::string name;
};

class Constant : public Diffusion {
public:
  Constant(const double eps) : Diffusion("CONSTANT"), epsilon(eps) {};
  void apply(mesh::Mesh *mesh) override;

private:
  const double epsilon;
};

} // namespace DIFF

#endif
