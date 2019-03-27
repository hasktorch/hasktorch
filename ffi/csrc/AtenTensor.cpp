#include<MacroPatternMatch.h>
#include "Mask.h"
#include "ATen/ATen.h"
#include "AtenTensor.h"

using namespace at;






TENSOR_DEF_VIRT(Tensor)

TENSOR_DEF_NONVIRT(Tensor)
TENSOR_DEF_ACCESSOR(Tensor)

