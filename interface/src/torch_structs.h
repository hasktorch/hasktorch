#include <stddef.h>

/*
  Concrete types
*/

typedef struct THAllocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void (*free)(void*, void*);
} THAllocator;

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397

typedef struct THGenerator {
  /* The initial seed. */
  unsigned long the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  unsigned long next;
  unsigned long state[_MERSENNE_STATE_N]; /* the array for the state vector  */
  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} THGenerator;

/*
  Templated generic types
*/

typedef struct THFloatStorage
{
  float *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THFloatStorage *view;
} THFloatStorage;

typedef struct THFloatTensor
{
  long *size;
  long *stride;
  int nDimension;
  THFloatStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THFloatTensor;


typedef struct THDoubleStorage
{
  double *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THDoubleStorage *view;
} THDoubleStorage;

typedef struct THDoubleTensor
{
  long *size;
  long *stride;
  int nDimension;
  THDoubleStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THDoubleTensor;

typedef struct THIntStorage
{
  int *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THIntStorage *view;
} THIntStorage;

typedef struct THIntTensor
{
  long *size;
  long *stride;
  int nDimension;
  THIntStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THIntTensor;

typedef struct THCharStorage
{
  char *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THCharStorage *view;
} THCharStorage;

typedef struct THCharTensor
{
  long *size;
  long *stride;
  int nDimension;
  THCharStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THCharTensor;

typedef struct THByteStorage
{
  unsigned char *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THByteStorage *view;
} THByteStorage;

typedef struct THByteTensor
{
  long *size;
  long *stride;
  int nDimension;
  THByteStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THByteTensor;
