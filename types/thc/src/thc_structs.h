#include <stddef.h>
// #include <cuda_runtime_api.h>

// See https://github.com/torch/cutorch/blob/master/FFI.lua
typedef struct cudaStream_t {} cudaStream_t;
/* struct cublasContext; */
typedef struct cublasHandle_t cublasHandle_t;
typedef struct cudaError_t {} cudaError_t;
// from /usr/local/cuda/cusparse.h
/* Opaque structure holding CUSPARSE library context */
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;


typedef struct THAllocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void (*free)(void*, void*);
} THAllocator;

// EVERYTHING ABOVE THIS SHOULD BE DELETED POST-C2HSC

// https://github.com/torch/cutorch/blob/79e393bade08b0090df8016bff56173a4a7f4845/lib/THC/THCStream.h
typedef struct THCStream {
  cudaStream_t stream;
  int device;
  int refcount;
} THCStream;




// https://github.com/torch/cutorch/blob/e2051b652d5b1f5182a1bfce11da9f53c2f92bd8/lib/THC/THCTensorRandom.h
/* Generator */
typedef struct _Generator {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
  int initf;
  unsigned long long initial_seed;
} Generator;

typedef struct THCRNGState {
  /* One generator per GPU */
  Generator* gen;
  int num_devices;
} THCRNGState;


// https://github.com/torch/cutorch/blob/ec93ff7b486274e248aa7156af7c8a1f16281e24/lib/THC/THCTensor.h
#define THC_DESC_BUFF_LEN 64
typedef struct THCDescBuff
{
  char str[THC_DESC_BUFF_LEN];
} THCDescBuff;



// https://github.com/torch/cutorch/blob/9f5cefdc86b3e5bf0344cf236231c9a82b3fcda8/lib/THC/THCGeneral.h.in
typedef struct _THCDeviceAllocator {
  cudaError_t (*malloc)( void*, void**, size_t,         cudaStream_t);
  cudaError_t (*realloc)(void*, void**, size_t, size_t, cudaStream_t);
  cudaError_t (*free)(void*, void*);
  cudaError_t (*emptyCache)(void*);
  cudaError_t (*cacheInfo)(void*, int, size_t*, size_t*);
  void* state;
} THCDeviceAllocator;

typedef struct _THCCudaResourcesPerDevice {
  THCStream** streams;
  /* Number of materialized cuBLAS handles */
  int numBlasHandles;
  /* Number of materialized cuSparse handles */
  int numSparseHandles;
  /* cuBLAS handes are lazily initialized */
  cublasHandle_t* blasHandles;
  /* cuSparse handes are lazily initialized */
  cusparseHandle_t* sparseHandles;
  /* Size of scratch space per each stream on this device available */
  size_t scratchSpacePerStream;
  /* Device-resident scratch space per stream, used for global memory
  *      reduction kernels. Lazily initialized. */
  void** devScratchSpacePerStream;
} THCCudaResourcesPerDevice;

/* Global state to be held in the cutorch table. */
typedef struct THCState {
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
 /* Set of all allocated resources. resourcePerDevice[dev]->streams[0] is NULL,
  * which specifies the per-device default stream. blasHandles and
  * sparseHandles do not have a default and must be explicitly initialized.
  * We always initialize 1 blasHandle and 1 sparseHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  /* Number of Torch defined resources available, indices 1 ... numStreams */
  int numUserStreams;
  int numUserBlasHandles;
  int numUserSparseHandles;

  /* Allocator using cudaMallocHost. */
  THAllocator* cudaHostAllocator;
  THAllocator* cudaUVAAllocator;
  THCDeviceAllocator* cudaDeviceAllocator;

 /* Index of the current selected BLAS handle. The actual BLAS handle used
  * depends on the current device. */
  // THCThreadLocal/*<int>*/ currentPerDeviceBlasHandle;                            // <================
 /* Index of the current selected sparse handle. The actual sparse handle used
  * depends on the current device. */
 // THCThreadLocal/*<int>*/ currentPerDeviceSparseHandle;                            // <================
  /* Array of thread locals containing the current stream for each device */
 // THCThreadLocal* currentStreams;                                                   // <================


 /* Table of enabled peer-to-peer access between directed pairs of GPUs.
  * If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;

 /*
  * Is direct cross-kernel p2p access allowed? Normally, only cross-GPU
  * copies are allowed via p2p if p2p access is enabled at all for
  * the pair of GPUs in question, but if this flag is true, then
  * all cross-GPU access checks are disabled, allowing kernels to
  * directly access memory on another GPUs.
  * Note that p2p access must exist and be enabled for the pair of
  * GPUs in question.
  */
  int p2pKernelAccessEnabled;

  void (*cutorchGCFunction)(void *data);
  void *cutorchGCData;
  ptrdiff_t heapSoftmax;
  ptrdiff_t heapDelta;
} THCState;

// https://github.com/torch/cutorch/blob/17300d9cc0c462dfde81eb81f89ba0a15e095844/lib/THC/generic/THCStorage.h
/*
 * typedef struct THCStorage
 * {
 *   real *data;
 *   ptrdiff_t size;
 *   int refcount;
 *   char flag;
 *   THCDeviceAllocator *allocator;
 *   void *allocatorContext;
 *   struct THCStorage *view;
 *   int device;
 * } THCStorage;
 */

// duplicated for each type:
typedef struct THCByteStorage
{
  unsigned char *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCByteStorage *view;
  int device;
} THCByteStorage;

typedef struct THCCharStorage
{
  char *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCCharStorage *view;
  int device;
} THCCharStorage;

typedef struct THCDoubleStorage
{
  double *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCDoubleStorage *view;
  int device;
} THCDoubleStorage;

typedef struct THCFloatStorage
{
  float *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCFloatStorage *view;
  int device;
} THCFloatStorage;

/*
 * typedef struct THCHalfStorage
 * {
 *   half *data;
 *   ptrdiff_t size;
 *   int refcount;
 *   char flag;
 *   THCDeviceAllocator *allocator;
 *   void *allocatorContext;
 *   struct THCHalfStorage *view;
 *   int device;
 * } THCHalfStorage;
 */


typedef struct THCIntStorage
{
  int *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCIntStorage *view;
  int device;
} THCIntStorage;

typedef struct THCShortStorage
{
  short *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCShortStorage *view;
  int device;
} THCShortStorage;

typedef struct THCLongStorage
{
  long *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
  struct THCLongStorage *view;
  int device;
} THCLongStorage;


// https://github.com/torch/cutorch/blob/9db5057877c6ffa7df59727cbada13318d7e3eaf/lib/THC/generic/THCTensor.h
/*
 * typedef struct THCTensor
 * {
 *   long *size;
 *   long *stride;
 *   int nDimension;
 *
 *   THCStorage *storage;
 *   ptrdiff_t storageOffset;
 *   int refcount;
 *
 *   char flag;
 *} THCTensor;
 */

// duplicated for each type:
typedef struct THCudaByteTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCByteStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaByteTensor;

typedef struct THCudaCharTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCCharStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaCharTensor;

typedef struct THCudaDoubleTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCDoubleStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaDoubleTensor;

typedef struct THCudaFloatTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCFloatStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaFloatTensor;

/*
 * typedef struct THCudaHalfTensor
 * {
 *   long *size;
 *   long *stride;
 *   int nDimension;
 *
 *   THCudaHalfStorage *storage;
 *   ptrdiff_t storageOffset;
 *   int refcount;
 *
 *   char flag;
 * } THCudaHalfTensor;
 */

typedef struct THCudaIntTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCIntStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaIntTensor;

typedef struct THCudaShortTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCShortStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaShortTensor;

typedef struct THCudaLongTensor
{
  long *size;
  long *stride;
  int nDimension;

  THCLongStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;

  char flag;
} THCudaLongTensor;

