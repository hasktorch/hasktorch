#include "hasktorch_finializer.h"
#undef CHECK
#include "Rts.h"

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


typedef void (*FUNC)(void *);
void
showCFinalizers(StgCFinalizerList *list)
{
  StgCFinalizerList *head;
  for (head = list;
       (StgClosure *)head != &stg_NO_FINALIZER_closure;
       head = (StgCFinalizerList *)head->link)
    {
      /* if (head->flag) */
      /* 	((void (*)(void *, void *))head->fptr)(head->eptr, head->ptr); */
      /* else */
      /* 	((void (*)(void *))head->fptr)(head->ptr); */
      FUNC ptr = (void (*)(void *))head->fptr;
      if(ptr == (FUNC)delete_tensor){
	printf("tensor\n");
      }
      if(ptr == (FUNC)delete_tensorlist){
	printf("tensorlist\n");
      }
      if(ptr == (FUNC)delete_tensorindex){
	printf("tensorindex\n");
      }
      if(ptr == (FUNC)delete_tensorindexlist){
	printf("tensorindexlist\n");
      }
      if(ptr == (FUNC)delete_c10dict){
	printf("c10dict\n");
      }
      if(ptr == (FUNC)delete_c10listivalue){
	printf("c10listivalue\n");
      }
      if(ptr == (FUNC)delete_c10listtensor){
	printf("c10listtensor\n");
      }
      if(ptr == (FUNC)delete_c10listdouble){
	printf("c10listdouble\n");
      }
      if(ptr == (FUNC)delete_c10listint){
	printf("c10listint\n");
      }
      if(ptr == (FUNC)delete_c10listbool){
	printf("c10listbool\n");
      }
      if(ptr == (FUNC)delete_stdvectordouble){
	printf("stdvectordouble\n");
      }
      if(ptr == (FUNC)delete_stdvectorint){
	printf("stdvectorint\n");
      }
      if(ptr == (FUNC)delete_stdvectorbool){
	printf("stdvectorbool\n");
      }
      if(ptr == (FUNC)delete_c10tuple){
	printf("c10tuple\n");
      }
      if(ptr == (FUNC)delete_context){
	printf("context\n");
      }
      if(ptr == (FUNC)delete_dimname){
	printf("dimname\n");
      }
      if(ptr == (FUNC)delete_dimnamelist){
	printf("dimnamelist\n");
      }
      if(ptr == (FUNC)delete_generator){
	printf("generator\n");
      }
      if(ptr == (FUNC)delete_ivalue){
	printf("ivalue\n");
      }
      if(ptr == (FUNC)delete_ivaluelist){
	printf("ivaluelist\n");
      }
      if(ptr == (FUNC)delete_intarray){
	printf("intarray\n");
      }
      if(ptr == (FUNC)delete_module){
	printf("module\n");
      }
      if(ptr == (FUNC)delete_jitgraph){
	printf("jitgraph\n");
      }
      if(ptr == (FUNC)delete_jitnode){
	printf("jitnode\n");
      }
      if(ptr == (FUNC)delete_jitvalue){
	printf("jitvalue\n");
      }
      if(ptr == (FUNC)delete_scalar){
	printf("scalar\n");
      }
      if(ptr == (FUNC)delete_stdarraybool2){
	printf("stdarraybool2\n");
      }
      if(ptr == (FUNC)delete_stdarraybool3){
	printf("stdarraybool3\n");
      }
      if(ptr == (FUNC)delete_stdarraybool4){
	printf("stdarraybool4\n");
      }
      if(ptr == (FUNC)delete_stdstring){
	printf("stdstring\n");
      }
      if(ptr == (FUNC)delete_storage){
	printf("storage\n");
      }
      if(ptr == (FUNC)delete_symbol){
	printf("symbol\n");
      }
      if(ptr == (FUNC)delete_tensoroptions){
	printf("tensoroptions\n");
      }
      if(ptr == (FUNC)delete_tensortensor){
	printf("tensorten\n");
      }
      if(ptr == (FUNC)delete_tensortensortensortensortensor){
	printf("tensortensortensortensorten\n");
      }
      if(ptr == (FUNC)delete_tensortensortensortensorlist){
	printf("tensortensortensortensorl\n");
      }
      if(ptr == (FUNC)delete_tensortensortensortensorint64){
	printf("tensortensortensortensorin\n");
      }
      if(ptr == (FUNC)delete_tensortensortensor){
	printf("tensortensorten\n");
      }
      if(ptr == (FUNC)delete_tensortensortensortensor){
	printf("tensortensortensorten\n");
      }
      if(ptr == (FUNC)delete_tensortensorcdoubleint64){
	printf("tensortensorcdoublein\n");
      }
      if(ptr == (FUNC)delete_cdoubleint64){
	printf("cdoublein\n");
      }
      if(ptr == (FUNC)delete_cdoublecdouble){
	printf("cdoublecdou\n");
      }
      if(ptr == (FUNC)delete_optimizer){
	printf("optimi\n");
      }
    }
}

void
showAllCFinalizers(StgWeak *list)
{
  StgWeak *w;
  for (w = list; w; w = w->link) {
    // We need to filter out DEAD_WEAK objects, because it's not guaranteed
    // that the list will not have them when shutting down.
    // They only get filtered out during GC for the generation they
    // belong to.
    // If there's no major GC between the time that the finalizer for the
    // object from the oldest generation is manually called and shutdown
    // we end up running the same finalizer twice. See #7170.
    const StgInfoTable *winfo = w->header.info;
    if (winfo != &stg_DEAD_WEAK_info) {
      showCFinalizers((StgCFinalizerList *)w->cfinalizers);
    }
  }

}

void
showWeakPtrList(){
  //  runAllCFinalizers(StgWeak *w)
  /* run C finalizers for all active weak pointers */
  //for (uint32_t i = 0; i < n_capabilities; i++) {
  // showAllCFinalizers(capabilities[i]->weak_ptr_list_hd);
  //}
  ACQUIRE_LOCK(sm_mutex);
  for (uint32_t g = 0; g < RtsFlags.GcFlags.generations; g++) {
    showAllCFinalizers(generations[g].weak_ptr_list);
  }
  RELEASE_LOCK(sm_mutex);
}
