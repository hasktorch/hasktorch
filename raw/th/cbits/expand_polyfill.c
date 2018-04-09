#include "TH/THTensor.h"
#include "TH/THStorage.h"

void THByteTensor_expand(THByteTensor *r, THByteTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THByteTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THByteTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THByteTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THByteTensor_setStorageNd(r, THByteTensor_storage(tensor), THByteTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THByteTensor_expand(THByteTensor *r, THByteTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THByteTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THByteTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THByteTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THByteTensor_setStorageNd(r, THByteTensor_storage(tensor), THByteTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THByteTensor_expandNd(THByteTensor **rets, THByteTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THByteTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THByteTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Byte: \n\n
void THCharTensor_expand(THCharTensor *r, THCharTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THCharTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THCharTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THCharTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THCharTensor_setStorageNd(r, THCharTensor_storage(tensor), THCharTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THCharTensor_expand(THCharTensor *r, THCharTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THCharTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THCharTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THCharTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THCharTensor_setStorageNd(r, THCharTensor_storage(tensor), THCharTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THCharTensor_expandNd(THCharTensor **rets, THCharTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THCharTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THCharTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Char: \n\n
void THShortTensor_expand(THShortTensor *r, THShortTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THShortTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THShortTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THShortTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THShortTensor_setStorageNd(r, THShortTensor_storage(tensor), THShortTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THShortTensor_expand(THShortTensor *r, THShortTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THShortTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THShortTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THShortTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THShortTensor_setStorageNd(r, THShortTensor_storage(tensor), THShortTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THShortTensor_expandNd(THShortTensor **rets, THShortTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THShortTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THShortTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Short: \n\n
void THIntTensor_expand(THIntTensor *r, THIntTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THIntTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THIntTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THIntTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THIntTensor_setStorageNd(r, THIntTensor_storage(tensor), THIntTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THIntTensor_expand(THIntTensor *r, THIntTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THIntTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THIntTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THIntTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THIntTensor_setStorageNd(r, THIntTensor_storage(tensor), THIntTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THIntTensor_expandNd(THIntTensor **rets, THIntTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THIntTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THIntTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Int: \n\n
void THLongTensor_expand(THLongTensor *r, THLongTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THLongTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THLongTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THLongTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THLongTensor_setStorageNd(r, THLongTensor_storage(tensor), THLongTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THLongTensor_expand(THLongTensor *r, THLongTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THLongTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THLongTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THLongTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THLongTensor_setStorageNd(r, THLongTensor_storage(tensor), THLongTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THLongTensor_expandNd(THLongTensor **rets, THLongTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THLongTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THLongTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Long: \n\n
void THHalfTensor_expand(THHalfTensor *r, THHalfTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THHalfTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THHalfTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THHalfTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THHalfTensor_setStorageNd(r, THHalfTensor_storage(tensor), THHalfTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THHalfTensor_expand(THHalfTensor *r, THHalfTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THHalfTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THHalfTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THHalfTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THHalfTensor_setStorageNd(r, THHalfTensor_storage(tensor), THHalfTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THHalfTensor_expandNd(THHalfTensor **rets, THHalfTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THHalfTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THHalfTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Half: \n\n
void THFloatTensor_expand(THFloatTensor *r, THFloatTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THFloatTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THFloatTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THFloatTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THFloatTensor_setStorageNd(r, THFloatTensor_storage(tensor), THFloatTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THFloatTensor_expand(THFloatTensor *r, THFloatTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THFloatTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THFloatTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THFloatTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THFloatTensor_setStorageNd(r, THFloatTensor_storage(tensor), THFloatTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THFloatTensor_expandNd(THFloatTensor **rets, THFloatTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THFloatTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THFloatTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Float: \n\n
void THDoubleTensor_expand(THDoubleTensor *r, THDoubleTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THDoubleTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THDoubleTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THDoubleTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THDoubleTensor_setStorageNd(r, THDoubleTensor_storage(tensor), THDoubleTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THDoubleTensor_expand(THDoubleTensor *r, THDoubleTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THDoubleTensor_nDimension(tensor) > 0, 0, "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THDoubleTensor_nDimension(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  long *expandedSizes;
  long *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THDoubleTensor_nDimension(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THDoubleTensor_setStorageNd(r, THDoubleTensor_storage(tensor), THDoubleTensor_storageOffset(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THDoubleTensor_expandNd(THDoubleTensor **rets, THDoubleTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THDoubleTensor_nDimension(ops[i]) > 0, i, "can't expand empty tensor %d", i);
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
    THDoubleTensor_expand(rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
}

\n\n\\ Polyfilling expand functions for Double: \n\n
