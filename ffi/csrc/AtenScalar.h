#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__Scalar__
#define __OUTPUT__Scalar__

#include "outputType.h"
 // 

 // 
#undef SCALAR_DECL_VIRT 
#define SCALAR_DECL_VIRT(Type) \


#undef SCALAR_DECL_NONVIRT 
#define SCALAR_DECL_NONVIRT(Type) \
Type ## _p Type ## _newScalar (  )

#undef SCALAR_DECL_ACCESSOR
#define SCALAR_DECL_ACCESSOR(Type)\


#undef SCALAR_DEF_VIRT
#define SCALAR_DEF_VIRT(Type)\


#undef SCALAR_DEF_NONVIRT
#define SCALAR_DEF_NONVIRT(Type)\
Type ## _p Type ## _newScalar (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef SCALAR_DEF_ACCESSOR
#define SCALAR_DEF_ACCESSOR(Type)\






SCALAR_DECL_VIRT(Scalar);


SCALAR_DECL_NONVIRT(Scalar);


SCALAR_DECL_ACCESSOR(Scalar);


#endif // __OUTPUT__Scalar__

#ifdef __cplusplus
}
#endif

