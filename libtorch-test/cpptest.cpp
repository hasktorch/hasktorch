#include <torch/torch.h>
#include <iostream>

// test wrapping

extern "C" {
  torch::Tensor wrap_rand(const int64_t dims[], const int ndim);
  torch::Tensor wrap_eye(const int64_t n);
  // FIXME: C ABI compatable return type
}

torch::Tensor wrap_rand(const int64_t dims[], const int ndim) {
  // int64_t dimsc[] = {4, 5};
  const c10::IntList dimsa = c10::ArrayRef<int64_t>(dims, ndim);
  return torch::rand(dimsa);
}

torch::Tensor wrap_eye(const int64_t n) {
  return at::eye(n);
}

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  const int64_t dims[3] = {4, 5, 2};
  std::cout << wrap_rand(dims, 3) << std::endl;
  std::cout << wrap_eye(3) << std::endl;

  auto a = torch::ones({4,3},torch::requires_grad());
  auto b = torch::ones({4,3},torch::requires_grad());
  auto c = at::add(a,b,1);
  std::cout << c[0][0] << std::endl;
  c.backward();
  std::cout << a.grad() << std::endl;
}
