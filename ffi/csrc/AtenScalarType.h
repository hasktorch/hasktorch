#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__ScalarType__
#define __OUTPUT__ScalarType__

#include "outputType.h"
 // 

 // 
#undef SCALARTYPE_DECL_VIRT 
#define SCALARTYPE_DECL_VIRT(Type) \


#undef SCALARTYPE_DECL_NONVIRT 
#define SCALARTYPE_DECL_NONVIRT(Type) \
Type ## _p Type ## _newScalarType (  )

#undef SCALARTYPE_DECL_ACCESSOR
#define SCALARTYPE_DECL_ACCESSOR(Type)\


#undef SCALARTYPE_DEF_VIRT
#define SCALARTYPE_DEF_VIRT(Type)\


#undef SCALARTYPE_DEF_NONVIRT
#define SCALARTYPE_DEF_NONVIRT(Type)\
Type ## _p Type ## _newScalarType (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef SCALARTYPE_DEF_ACCESSOR
#define SCALARTYPE_DEF_ACCESSOR(Type)\






SCALARTYPE_DECL_VIRT(ScalarType);


SCALARTYPE_DECL_NONVIRT(ScalarType);


SCALARTYPE_DECL_ACCESSOR(ScalarType);


#endif // __OUTPUT__ScalarType__

#ifdef __cplusplus
}
#endif

