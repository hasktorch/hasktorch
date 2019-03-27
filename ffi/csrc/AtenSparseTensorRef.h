#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__SparseTensorRef__
#define __OUTPUT__SparseTensorRef__

#include "outputType.h"
 // 

 // 
#undef SPARSETENSORREF_DECL_VIRT 
#define SPARSETENSORREF_DECL_VIRT(Type) \


#undef SPARSETENSORREF_DECL_NONVIRT 
#define SPARSETENSORREF_DECL_NONVIRT(Type) \
Type ## _p Type ## _newSparseTensorRef ( Tensor_p t )

#undef SPARSETENSORREF_DECL_ACCESSOR
#define SPARSETENSORREF_DECL_ACCESSOR(Type)\


#undef SPARSETENSORREF_DEF_VIRT
#define SPARSETENSORREF_DEF_VIRT(Type)\


#undef SPARSETENSORREF_DEF_NONVIRT
#define SPARSETENSORREF_DEF_NONVIRT(Type)\
Type ## _p Type ## _newSparseTensorRef ( Tensor_p t )\
{\
Type * newp = new Type (to_nonconstref<Tensor,Tensor_t>(*t)); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef SPARSETENSORREF_DEF_ACCESSOR
#define SPARSETENSORREF_DEF_ACCESSOR(Type)\






SPARSETENSORREF_DECL_VIRT(SparseTensorRef);


SPARSETENSORREF_DECL_NONVIRT(SparseTensorRef);


SPARSETENSORREF_DECL_ACCESSOR(SparseTensorRef);


#endif // __OUTPUT__SparseTensorRef__

#ifdef __cplusplus
}
#endif

