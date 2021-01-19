#include "hasktorch_finializer.h"

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

void delete_cdoubleint64(std::tuple<double,int64_t>* object){
  delete object;
}

void delete_cdoublecdouble(std::tuple<double,double>* object){
  delete object;
}

void delete_optimizer(torch::optim::Optimizer* object){
  delete object;
}

void
showObject(void* ptr, void* fptr){
  if(fptr == (void*)delete_tensor){
    at::Tensor* t = (at::Tensor*) ptr;
    std::cout << "Tensor:" << t->sizes() << std::endl;
  }
  if(fptr == (void*)delete_tensorlist){
    printf("tensorlist\n");
  }
  if(fptr == (void*)delete_tensorindex){
    printf("tensorindex\n");
  }
  if(fptr == (void*)delete_tensorindexlist){
    printf("tensorindexlist\n");
  }
  if(fptr == (void*)delete_c10dict){
    printf("c10dict\n");
  }
  if(fptr == (void*)delete_c10listivalue){
    printf("c10listivalue\n");
  }
  if(fptr == (void*)delete_c10listtensor){
    printf("c10listtensor\n");
  }
  if(fptr == (void*)delete_c10listdouble){
    printf("c10listdouble\n");
  }
  if(fptr == (void*)delete_c10listint){
    printf("c10listint\n");
  }
  if(fptr == (void*)delete_c10listbool){
    printf("c10listbool\n");
  }
  if(fptr == (void*)delete_stdvectordouble){
    printf("stdvectordouble\n");
  }
  if(fptr == (void*)delete_stdvectorint){
    printf("stdvectorint\n");
  }
  if(fptr == (void*)delete_stdvectorbool){
    printf("stdvectorbool\n");
  }
  if(fptr == (void*)delete_c10tuple){
    printf("c10tuple\n");
  }
  if(fptr == (void*)delete_context){
    printf("context\n");
  }
  if(fptr == (void*)delete_dimname){
    printf("dimname\n");
  }
  if(fptr == (void*)delete_dimnamelist){
    printf("dimnamelist\n");
  }
  if(fptr == (void*)delete_generator){
    printf("generator\n");
  }
  if(fptr == (void*)delete_ivalue){
    printf("ivalue\n");
  }
  if(fptr == (void*)delete_ivaluelist){
    printf("ivaluelist\n");
  }
  if(fptr == (void*)delete_intarray){
    printf("intarray\n");
  }
  if(fptr == (void*)delete_module){
    printf("module\n");
  }
  if(fptr == (void*)delete_jitgraph){
    printf("jitgraph\n");
  }
  if(fptr == (void*)delete_jitnode){
    printf("jitnode\n");
  }
  if(fptr == (void*)delete_jitvalue){
    printf("jitvalue\n");
  }
  if(fptr == (void*)delete_scalar){
    printf("scalar\n");
  }
  if(fptr == (void*)delete_stdarraybool2){
    printf("stdarraybool2\n");
  }
  if(fptr == (void*)delete_stdarraybool3){
    printf("stdarraybool3\n");
  }
  if(fptr == (void*)delete_stdarraybool4){
    printf("stdarraybool4\n");
  }
  if(fptr == (void*)delete_stdstring){
    printf("stdstring\n");
  }
  if(fptr == (void*)delete_storage){
    printf("storage\n");
  }
  if(fptr == (void*)delete_symbol){
    printf("symbol\n");
  }
  if(fptr == (void*)delete_tensoroptions){
    printf("tensoroptions\n");
  }
  if(fptr == (void*)delete_tensortensor){
    printf("tensorten\n");
  }
  if(fptr == (void*)delete_tensortensortensortensortensor){
    printf("tensortensortensortensorten\n");
  }
  if(fptr == (void*)delete_tensortensortensortensorlist){
    printf("tensortensortensortensorl\n");
  }
  if(fptr == (void*)delete_tensortensortensortensorint64){
    printf("tensortensortensortensorin\n");
  }
  if(fptr == (void*)delete_tensortensortensor){
    printf("tensortensorten\n");
  }
  if(fptr == (void*)delete_tensortensortensortensor){
    printf("tensortensortensorten\n");
  }
  if(fptr == (void*)delete_tensortensorcdoubleint64){
    printf("tensortensorcdoublein\n");
  }
  if(fptr == (void*)delete_cdoubleint64){
    printf("cdoublein\n");
  }
  if(fptr == (void*)delete_cdoublecdouble){
    printf("cdoublecdou\n");
  }
  if(fptr == (void*)delete_optimizer){
    printf("optimi\n");
  }
}

