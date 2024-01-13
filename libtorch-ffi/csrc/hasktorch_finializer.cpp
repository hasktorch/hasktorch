#include "hasktorch_finializer.h"
#include <map>

void delete_tensor(at::Tensor* tensor){
  delete tensor;
}

void delete_tensorlist(std::vector<at::Tensor>* tensors){
  delete tensors;
}

void delete_tensorindex(at::indexing::TensorIndex* idx){
  delete idx;
}

void delete_tensorindexlist(std::vector<at::indexing::TensorIndex>* idxs){
  delete idxs;
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

void delete_c10listoptionaltensor(c10::List<c10::optional<at::Tensor>>* object){
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

void delete_stdvectordouble(std::vector<double>* object){
  delete object;
}

void delete_stdvectorint(std::vector<int64_t>* object){
  delete object;
}

void delete_stdvectorbool(std::vector<bool>* object){
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

void delete_tensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensortensortensortensorint64int64int64int64tensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,at::Tensor,at::Tensor>* object){
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

void delete_tensortensorint64int64tensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor>* object){
  delete object;
}

void delete_tensortensorint64int64tensortensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor>* object){
  delete object;
}

void delete_tensorlisttensor(std::tuple<std::vector<at::Tensor>,at::Tensor>* object){
  delete object;
}

void delete_tensortensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>>* object){
  delete object;
}

void delete_tensortensorlisttensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor>>* object){
  delete object;
}

void delete_tensorlisttensorlisttensorlisttensorlisttensorlist(std::tuple<std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>>* object){
  delete object;
}

void delete_cdoubleint64(std::tuple<double,int64_t>* object){
  delete object;
}

void delete_cdoublecdouble(std::tuple<double,double>* object){
  delete object;
}

void delete_tensorgenerator(std::tuple<at::Tensor,at::Generator>* object){
  delete object;
}

void delete_optimizer(torch::optim::Optimizer* object){
  delete object;
}

void delete_stream(c10::Stream* object){
  delete object;
}

void delete_arrayrefscalar(at::ArrayRef<at::Scalar>* object){
  delete object;
}

void delete_vectorscalar(std::vector<at::Scalar>* object){
  delete object;
}

std::map<void*,int> objectAge;
std::map<void*,int> prevObjectAge;

void
shiftObjectMap(){
  prevObjectAge = objectAge;
  objectAge = std::map<void*,int>();
}

void
showObject(int flag, void* ptr, void* fptr){
  auto it = prevObjectAge.find(ptr);
  int age = 0;
  if (it != prevObjectAge.end()) {
    objectAge[ptr] = it->second + 1;
    age = it->second + 1;
  } else {
    objectAge[ptr] = 1;
    age = 1;
  }
  if(flag == 0)
    return;
  if(age < flag)
    return;
  if(fptr == (void*)delete_tensor){
    at::Tensor* t = (at::Tensor*) ptr;
    std::cout << age << ":" << "Tensor " << t->scalar_type() << " " << t->sizes() << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlist){
    std::cout << age << ":" << "[Tensor]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorindex){
    std::cout << age << ":" << "tensorindex" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorindexlist){
    std::cout << age << ":" << "[tensorindex]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10dict){
    std::cout << age << ":" << "c10dict" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listivalue){
    std::cout << age << ":" << "c10listivalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listtensor){
    std::cout << age << ":" << "c10listtensor" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listoptionaltensor){
    std::cout << age << ":" << "c10listoptionaltensor" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listdouble){
    std::cout << age << ":" << "c10listdouble" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listint){
    std::cout << age << ":" << "c10listint" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listbool){
    std::cout << age << ":" << "c10listbool" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectordouble){
    std::cout << age << ":" << "std::vector<double>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectorint){
    std::cout << age << ":" << "std::vector<int>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectorbool){
    std::cout << age << ":" << "std::vector<bool>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10tuple){
    std::cout << age << ":" << "c10tuple" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_context){
    std::cout << age << ":" << "context" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_dimname){
    std::cout << age << ":" << "dimname" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_dimnamelist){
    std::cout << age << ":" << "[dimname]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_generator){
    std::cout << age << ":" << "generator" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_ivalue){
    std::cout << age << ":" << "ivalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_ivaluelist){
    std::cout << age << ":" << "[ivalue]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_intarray){
    std::cout << age << ":" << "intarray" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_module){
    std::cout << age << ":" << "module" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitgraph){
    std::cout << age << ":" << "jitgraph" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitnode){
    std::cout << age << ":" << "jitnode" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitvalue){
    std::cout << age << ":" << "jitvalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_scalar){
    std::cout << age << ":" << "scalar" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool2){
    std::cout << age << ":" << "std::array<bool,2>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool3){
    std::cout << age << ":" << "std::array<bool,3>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool4){
    std::cout << age << ":" << "std::array<bool,4>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdstring){
    std::cout << age << ":" << "std::string" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_storage){
    std::cout << age << ":" << "storage" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_symbol){
    std::cout << age << ":" << "symbol" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensoroptions){
    std::cout << age << ":" << "tensoroptions" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensor){
    std::cout << age << ":" << "(tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorint64int64int64int64tensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,int,int,int,int,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorlist){
    std::cout << age << ":" << "(tensor,tensor,tensor,[tensor])" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorint64){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorcdoubleint64){
    std::cout << age << ":" << "(tensor,tensor,double,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorint64int64tensor){
    std::cout << age << ":" << "(tensor,tensor,int,int,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorint64int64tensortensor){
    std::cout << age << ":" << "(tensor,tensor,int,int,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlisttensor){
    std::cout << age << ":" << "(tensorlist,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorlist){
    std::cout << age << ":" << "(tensor,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorlisttensorlist){
    std::cout << age << ":" << "(tensor,tensorlist,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlisttensorlisttensorlisttensorlisttensorlist){
    std::cout << age << ":" << "(tensorlist,tensorlist,tensorlist,tensorlist,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_cdoubleint64){
    std::cout << age << ":" << "(double,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_cdoublecdouble){
    std::cout << age << ":" << "(double,double)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorgenerator){
    std::cout << age << ":" << "(tensor,generator)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_optimizer){
    std::cout << age << ":" << "optimizer" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stream){
    std::cout << age << ":" << "stream" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_arrayrefscalar){
    std::cout << age << ":" << "at::ArrayRef<at::Scalar>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_vectorscalar){
    std::cout << age << ":" << "std::vector<at::Scalar>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }
}

