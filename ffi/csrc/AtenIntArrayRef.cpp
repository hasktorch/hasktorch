#include<MacroPatternMatch.h>
#include "Mask.h"
#include "ATen/ATen.h"
#include "AtenIntArrayRef.h"

using namespace at;






INTARRAYREF_DEF_VIRT(IntArrayRef)

INTARRAYREF_DEF_NONVIRT(IntArrayRef)
INTARRAYREF_DEF_ACCESSOR(IntArrayRef)

