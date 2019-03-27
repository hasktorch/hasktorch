#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__Tensor__
#define __OUTPUT__Tensor__

#include "outputType.h"
 // 

 // 
#undef TENSOR_DECL_VIRT 
#define TENSOR_DECL_VIRT(Type) \


#undef TENSOR_DECL_NONVIRT 
#define TENSOR_DECL_NONVIRT(Type) \
Type ## _p Type ## _newTensor (  ); \
signed long Type ## _tensor_dim ( Type ## _p p ); \
signed long Type ## _tensor_storage_offset ( Type ## _p p ); \
int Type ## _tensor_defined ( Type ## _p p ); \
void Type ## _tensor_reset ( Type ## _p p ); \
void Type ## _tensor_cpu ( Type ## _p p ); \
void Type ## _tensor_cuda ( Type ## _p p ); \
void Type ## _tensor_print ( Type ## _p p )

#undef TENSOR_DECL_ACCESSOR
#define TENSOR_DECL_ACCESSOR(Type)\


#undef TENSOR_DEF_VIRT
#define TENSOR_DEF_VIRT(Type)\


#undef TENSOR_DEF_NONVIRT
#define TENSOR_DEF_NONVIRT(Type)\
Type ## _p Type ## _newTensor (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}\
signed long Type ## _tensor_dim ( Type ## _p p )\
{\
return to_nonconst<Type,Type ## _t>(p)->dim();\
}\
signed long Type ## _tensor_storage_offset ( Type ## _p p )\
{\
return to_nonconst<Type,Type ## _t>(p)->storage_offset();\
}\
int Type ## _tensor_defined ( Type ## _p p )\
{\
return to_nonconst<Type,Type ## _t>(p)->defined();\
}\
void Type ## _tensor_reset ( Type ## _p p )\
{\
to_nonconst<Type,Type ## _t>(p)->reset();\
}\
void Type ## _tensor_cpu ( Type ## _p p )\
{\
to_nonconst<Type,Type ## _t>(p)->cpu();\
}\
void Type ## _tensor_cuda ( Type ## _p p )\
{\
to_nonconst<Type,Type ## _t>(p)->cuda();\
}\
void Type ## _tensor_print ( Type ## _p p )\
{\
to_nonconst<Type,Type ## _t>(p)->print();\
}

#undef TENSOR_DEF_ACCESSOR
#define TENSOR_DEF_ACCESSOR(Type)\






TENSOR_DECL_VIRT(Tensor);


TENSOR_DECL_NONVIRT(Tensor);


TENSOR_DECL_ACCESSOR(Tensor);


#endif // __OUTPUT__Tensor__

#ifdef __cplusplus
}
#endif

