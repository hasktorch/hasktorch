#include<MacroPatternMatch.h>
#include "AtenTensor.h"
#include "Mask.h"
#include "ATen/ATen.h"
#include "AtenSparseTensorRef.h"

using namespace at;






SPARSETENSORREF_DEF_VIRT(SparseTensorRef)

SPARSETENSORREF_DEF_NONVIRT(SparseTensorRef)
SPARSETENSORREF_DEF_ACCESSOR(SparseTensorRef)

