#pragma once
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
Type ## _p Type ## _newTensor (  )

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

