#ifdef __cplusplus
extern "C" { 
#endif

#ifndef __OUTPUT__TensorOptions__
#define __OUTPUT__TensorOptions__

#include "outputType.h"
 // 

 // 
#undef TENSOROPTIONS_DECL_VIRT 
#define TENSOROPTIONS_DECL_VIRT(Type) \


#undef TENSOROPTIONS_DECL_NONVIRT 
#define TENSOROPTIONS_DECL_NONVIRT(Type) \
Type ## _p Type ## _newTensorOptions (  )

#undef TENSOROPTIONS_DECL_ACCESSOR
#define TENSOROPTIONS_DECL_ACCESSOR(Type)\


#undef TENSOROPTIONS_DEF_VIRT
#define TENSOROPTIONS_DEF_VIRT(Type)\


#undef TENSOROPTIONS_DEF_NONVIRT
#define TENSOROPTIONS_DEF_NONVIRT(Type)\
Type ## _p Type ## _newTensorOptions (  )\
{\
Type * newp = new Type (); \
return to_nonconst<Type ## _t, Type >(newp);\
}

#undef TENSOROPTIONS_DEF_ACCESSOR
#define TENSOROPTIONS_DEF_ACCESSOR(Type)\






TENSOROPTIONS_DECL_VIRT(TensorOptions);


TENSOROPTIONS_DECL_NONVIRT(TensorOptions);


TENSOROPTIONS_DECL_ACCESSOR(TensorOptions);


#endif // __OUTPUT__TensorOptions__

#ifdef __cplusplus
}
#endif

