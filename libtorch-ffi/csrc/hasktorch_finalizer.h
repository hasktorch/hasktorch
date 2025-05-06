#include <ATen/Tensor.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/Dimname.h>
#include <ATen/Storage.h>
#include <ATen/TensorIndexing.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/optim.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

extern "C" {
  void delete_tensor(at::Tensor* tensor);

  void delete_tensorlist(std::vector<at::Tensor>* tensors);

  void delete_tensorindex(at::indexing::TensorIndex* idx);

  void delete_tensorindexlist(std::vector<at::indexing::TensorIndex>* idxs);

  void delete_c10dict(c10::Dict<at::IValue,at::IValue>* object);

  void delete_c10listivalue(c10::List<at::IValue>* object);

  void delete_c10listtensor(c10::List<at::Tensor>* object);

  void delete_c10listoptionaltensor(c10::List<c10::optional<at::Tensor>>* object);

  void delete_c10listdouble(c10::List<double>* object);

  void delete_c10listint(c10::List<int64_t>* object);

  void delete_c10listbool(c10::List<bool>* object);

  void delete_stdvectordouble(std::vector<double>* object);

  void delete_stdvectorint(std::vector<int64_t>* object);

  void delete_stdvectorbool(std::vector<bool>* object);

  void delete_c10tuple(c10::intrusive_ptr<at::ivalue::Tuple>* object);

  void delete_context(at::Context* object);

  void delete_dimname(at::Dimname* object);

  void delete_dimnamelist(std::vector<at::Dimname>* object);

  void delete_generator(at::Generator* object);

  void delete_ivalue(at::IValue* object);

  void delete_ivaluelist(std::vector<at::IValue>* object);

  void delete_intarray(std::vector<int64_t>* object);

  void delete_module(torch::jit::script::Module* object);

  void delete_jitgraph(std::shared_ptr<torch::jit::Graph>* object);

  void delete_jitnode(torch::jit::Node* object);

  void delete_jitvalue(torch::jit::Value* object);

  void delete_scalar(at::Scalar* object);

  void delete_stdarraybool2(std::array<bool,2>* object);

  void delete_stdarraybool3(std::array<bool,3>* object);

  void delete_stdarraybool4(std::array<bool,4>* object);

  void delete_stdstring(std::string* object);

  void delete_storage(at::Storage* object);

  void delete_symbol(at::Symbol* object);

  void delete_tensoroptions(at::TensorOptions* object);

  void delete_tensortensor(std::tuple<at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensorint64int64int64int64tensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensorlist(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* ptr);

  void delete_tensortensortensortensorint64(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>* ptr);

  void delete_tensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr);

  void delete_tensortensorcdoubleint64(std::tuple<at::Tensor,at::Tensor,double,int64_t>* ptr);

  void delete_tensortensorint64int64tensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor>* ptr);

  void delete_tensortensorint64int64tensortensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor>* ptr);

  void delete_tensorlisttensor(std::tuple<std::vector<at::Tensor>,at::Tensor>* ptr);

  void delete_tensortensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>>* ptr);

  void delete_tensortensorlisttensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor>>* ptr);

  void delete_tensorlisttensorlisttensorlisttensorlisttensorlist(std::tuple<std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>>* ptr);

  void delete_cdoubleint64(std::tuple<double,int64_t>* ptr);

  void delete_cdoublecdouble(std::tuple<double,double>* ptr);

  void delete_tensorgenerator(std::tuple<at::Tensor,at::Generator>* ptr);

  void delete_optimizer(torch::optim::Optimizer* ptr);

  void delete_stream(c10::Stream* ptr);

  void delete_arrayrefscalar(at::ArrayRef<at::Scalar>* ptr);

  void delete_vectorscalar(std::vector<at::Scalar>* ptr);

#include "hasktorch_dump.h"
};
