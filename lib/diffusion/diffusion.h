#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "../space/mesh.h"
#include <string>

#ifdef WITH_ONNX
#include <memory>
#include <vector>
#include <onnxruntime_cxx_api.h>
#endif

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

class Custom : public Diffusion {
public:
  Custom(const double *eps_array)
      : Diffusion("CUSTOM"), eps_array(eps_array) {};
  void apply(mesh::Mesh *mesh) override;

private:
  const double *eps_array;
};

#ifdef WITH_ONNX
/**
 * @brief Diffusion driven by a trained ONNX model.
 * Input : [rho | rhou | e] concatenated  (1 x 3*n_total float32)
 * Output: eps per node                   (1 x n_total   float32, >= 0)
 */
class ONNX : public Diffusion {
public:
  ONNX(const std::string &model_path, int n_total);
  void apply(mesh::Mesh *mesh) override;

private:
  int n_total;
  std::vector<float>  input_buf;
  std::vector<float>  output_buf;
  std::vector<double> eps_buf;       // float->double conversion for diffuse()
  std::unique_ptr<Ort::Env>     env;
  std::unique_ptr<Ort::Session> session;
};
#endif // WITH_ONNX

} // namespace DIFF

#endif
