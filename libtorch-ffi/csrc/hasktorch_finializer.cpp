#include "hasktorch_finializer.h"

void delete_tensor(at::Tensor* tensor){
  delete tensor;
}

void delete_tensorlist(std::vector<at::Tensor>* tensors){
  delete tensors;
}

void delete_c10dict(c10::Dict<at::IValue,at::IValue>* object){
  delete object;
}

void delete_c10listivalue(c10::List<at::IValue>* object){
  delete object;
}

void delete_c10listtensor(c10::List<at::Tensor>* object){
  delete object;
}

void delete_c10listdouble(c10::List<double>* object){
  delete object;
}

void delete_c10listint(c10::List<int64_t>* object){
  delete object;
}

void delete_c10listbool(c10::List<bool>* object){
  delete object;
}

void delete_c10tuple(c10::intrusive_ptr<at::ivalue::Tuple>* object){
  delete object;
}

void delete_context(at::Context* object){
  delete object;
}

void delete_dimname(at::Dimname* object){
  delete object;
}

void delete_dimnamelist(std::vector<at::Dimname>* object){
  delete object;
}

void delete_generator(at::Generator* object){
  delete object;
}

void delete_ivalue(at::IValue* object){
  delete object;
}

void delete_ivaluelist(std::vector<at::IValue>* object){
  delete object;
}

void delete_intarray(std::vector<int64_t>* object){
  delete object;
}

void delete_module(torch::jit::script::Module* object){
  delete object;
}

void delete_jitgraph(std::shared_ptr<torch::jit::Graph>* object){
  delete object;
}

void delete_jitnode(torch::jit::Node* object){
  delete object;
}

void delete_jitvalue(torch::jit::Value* object){
  delete object;
}

void delete_scalar(at::Scalar* object){
  delete object;
}

void delete_stdarraybool2(std::array<bool,2>* object){
  delete object;
}

void delete_stdarraybool3(std::array<bool,3>* object){
  delete object;
}

void delete_stdarraybool4(std::array<bool,4>* object){
  delete object;
}

void delete_stdstring(std::string* object){
  delete object;
}

void delete_storage(at::Storage* object){
  delete object;
}

void delete_symbol(at::Symbol* object){
  delete object;
}

void delete_tensoroptions(at::TensorOptions* object){
  delete object;
}

void delete_tensortensor(std::tuple<at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensortensortensorlist(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* object){
  delete object;
}

void delete_tensortensortensortensorint64(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>* object){
  delete object;
}

void delete_tensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensorcdoubleint64(std::tuple<at::Tensor,at::Tensor,double,int64_t>* object){
  delete object;
}
