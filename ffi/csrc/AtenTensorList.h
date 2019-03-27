#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__TensorList__
#define __OUTPUT__TensorList__

#include "outputType.h"
 // 

 // 
#undef TENSORLIST_DECL_VIRT 
#define TENSORLIST_DECL_VIRT(Type) \


#undef TENSORLIST_DECL_NONVIRT 
#define TENSORLIST_DECL_NONVIRT(Type) \
Type ## _p Type ## _newTensorList (  )

#undef TENSORLIST_DECL_ACCESSOR
#define TENSORLIST_DECL_ACCESSOR(Type)\


#undef TENSORLIST_DEF_VIRT
#define TENSORLIST_DEF_VIRT(Type)\


#undef TENSORLIST_DEF_NONVIRT
#define TENSORLIST_DEF_NONVIRT(Type)\
Type ## _p Type ## _newTensorList (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef TENSORLIST_DEF_ACCESSOR
#define TENSORLIST_DEF_ACCESSOR(Type)\






TENSORLIST_DECL_VIRT(TensorList);


TENSORLIST_DECL_NONVIRT(TensorList);


TENSORLIST_DECL_ACCESSOR(TensorList);


#endif // __OUTPUT__TensorList__

#ifdef __cplusplus
}
#endif

