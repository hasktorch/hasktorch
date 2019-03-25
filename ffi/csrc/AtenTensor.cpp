#include <ATen/ATen.h>

// Both c10/util/logging_is_not_google_glog.h and MacroPatternMatch.h define 'CHECK'.
// The conflict of CHECK-define causes compile error.
// To prevent the error, undef CHECK-define.
#undef CHECK
#include <MacroPatternMatch.h>

#include "AtenTensor.h"


using namespace at;




TENSOR_DEF_VIRT(Tensor)

TENSOR_DEF_NONVIRT(Tensor)
TENSOR_DEF_ACCESSOR(Tensor)

