#include "THC/THCStorage.h"
#include "THC/THCTensor.h"

THCTensor* THCTensor_(newExpand)(THCState *state, THCTensor *tensor, THLongStorage *sizes)
{
  THCTensor *result = THCTensor_(new)(state);
  THCTensor_(expand)(state, result, tensor, sizes);
  return result;
}

void THCTensor_(expand)(THCState *state, THCTensor *r, THCTensor *tensor, THLongStorage *sizes)
{
  THArgCheck(THCTensor_(nDimension)(state, tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THCTensor_(nDimension)(state, tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret = THLongStorage_inferExpandGeometry(tensor->size,
                                              tensor->stride,
                                              THCTensor_(nDimension)(state, tensor),
                                              sizes,
                                              &expandedSizes,
                                              &expandedStrides,
                                              error_buffer,
                                              1024);
  if (ret != 0) {
    THError(error_buffer);
    return;
  }
  THCTensor_(setStorageNd)(state, r, THCTensor_(storage)(state, tensor), THCTensor_(storageOffset)(state, tensor),
                           THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THCTensor_(expandNd)(THCState *state, THCTensor **rets, THCTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THCTensor_(nDimension)(state, ops[i]) > 0, i, "can't expand empty tensor %d", i);
  }

  long *op_sizes[count];
  long op_dims[count];

  for (int i = 0; i < count; ++i) {
    op_sizes[i] = ops[i]->size;
    op_dims[i] = ops[i]->nDimension;
  }

  THLongStorage *sizes = THLongStorage_new();
  char error_buffer[1024];
  int ret = THLongStorage_inferSizeN(sizes,
                                     count,
                                     op_sizes,
                                     op_dims,
                                     error_buffer,
                                     1024);

  if(ret != 0) {
    THLongStorage_free(sizes);
    THError(error_buffer);
    return;
  }

  for (int i = 0; i < count; ++i) {
    THCTensor_(expand)(state, rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

