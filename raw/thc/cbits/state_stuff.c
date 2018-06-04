#ifdef CUDA
#include "THC/THCGeneral.h"
#endif
void free_CTHState(
#ifdef CUDA
  THCState* s
#else
  int* s
#endif
  ) { return; }
