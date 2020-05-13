#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/script.h> // One-stop header.
#include <vector>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: alexnet <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<at::Tensor> vec_parameters;
  auto parameters = module.parameters();
  for (auto p : parameters)
  {
    vec_parameters.push_back(p);
  }
  
  const std::string parameters_save_path = "alexnet.pt";

  // Save the parameters
  torch::save(vec_parameters, parameters_save_path);
}
