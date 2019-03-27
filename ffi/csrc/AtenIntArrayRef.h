#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__IntArrayRef__
#define __OUTPUT__IntArrayRef__

#include "outputType.h"
 // 

 // 
#undef INTARRAYREF_DECL_VIRT 
#define INTARRAYREF_DECL_VIRT(Type) \


#undef INTARRAYREF_DECL_NONVIRT 
#define INTARRAYREF_DECL_NONVIRT(Type) \
Type ## _p Type ## _newIntArrayRef (  )

#undef INTARRAYREF_DECL_ACCESSOR
#define INTARRAYREF_DECL_ACCESSOR(Type)\


#undef INTARRAYREF_DEF_VIRT
#define INTARRAYREF_DEF_VIRT(Type)\


#undef INTARRAYREF_DEF_NONVIRT
#define INTARRAYREF_DEF_NONVIRT(Type)\
Type ## _p Type ## _newIntArrayRef (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef INTARRAYREF_DEF_ACCESSOR
#define INTARRAYREF_DEF_ACCESSOR(Type)\






INTARRAYREF_DECL_VIRT(IntArrayRef);


INTARRAYREF_DECL_NONVIRT(IntArrayRef);


INTARRAYREF_DECL_ACCESSOR(IntArrayRef);


#endif // __OUTPUT__IntArrayRef__

#ifdef __cplusplus
}
#endif

