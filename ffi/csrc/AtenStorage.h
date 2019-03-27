#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__Storage__
#define __OUTPUT__Storage__

#include "outputType.h"
 // 

 // 
#undef STORAGE_DECL_VIRT 
#define STORAGE_DECL_VIRT(Type) \


#undef STORAGE_DECL_NONVIRT 
#define STORAGE_DECL_NONVIRT(Type) \
Type ## _p Type ## _newStorage (  )

#undef STORAGE_DECL_ACCESSOR
#define STORAGE_DECL_ACCESSOR(Type)\


#undef STORAGE_DEF_VIRT
#define STORAGE_DEF_VIRT(Type)\


#undef STORAGE_DEF_NONVIRT
#define STORAGE_DEF_NONVIRT(Type)\
Type ## _p Type ## _newStorage (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef STORAGE_DEF_ACCESSOR
#define STORAGE_DEF_ACCESSOR(Type)\






STORAGE_DECL_VIRT(Storage);


STORAGE_DECL_NONVIRT(Storage);


STORAGE_DECL_ACCESSOR(Storage);


#endif // __OUTPUT__Storage__

#ifdef __cplusplus
}
#endif

