#ifdef CUDA
#include "THC/THCGeneral.h"
void free_CTHState(THCState* s)
{
  THCudaShutdown(s);
  THCState_free(s);
}
#else
void free_CTHState(int* s)
{
  return;
}
#endif
