{-# OPTIONS_GHC -fno-warn-unused-imports #-}
#include "bindings.dsl.h"
#include "torch_structs.h"
module TorchStructs where
import Foreign.Ptr
#strict_import

{- typedef struct THAllocator {
            void * (* malloc)(void *, ptrdiff_t);
            void * (* realloc)(void *, void *, ptrdiff_t);
            void (* free)(void *, void *);
        } THAllocator; -}
#starttype struct THAllocator
#field malloc , FunPtr (Ptr () -> CLong -> Ptr ())
#field realloc , FunPtr (Ptr () -> Ptr () -> CLong -> Ptr ())
#field free , FunPtr (Ptr () -> Ptr () -> IO ())
#stoptype

{- typedef struct THGenerator {
            unsigned long the_initial_seed;
            int left;
            int seeded;
            unsigned long next;
            unsigned long state[624];
            double normal_x;
            double normal_y;
            double normal_rho;
            int normal_is_valid;
        } THGenerator; -}
#starttype struct THGenerator
#field the_initial_seed , CULong
#field left , CInt
#field seeded , CInt
#field next , CULong
#array_field state , CULong
#field normal_x , CDouble
#field normal_y , CDouble
#field normal_rho , CDouble
#field normal_is_valid , CInt
#stoptype

{- typedef struct THFloatStorage {
            float * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THFloatStorage * view;
        } THFloatStorage; -}
#starttype struct THFloatStorage
#field data , Ptr CFloat
#field size , CLong
#field refcount , CInt
#field flag , CChar
#field allocator , Ptr <struct THAllocator>
#field allocatorContext , Ptr ()
#field view , Ptr <struct THFloatStorage>
#stoptype

{- typedef struct THFloatTensor {
            long * size;
            long * stride;
            int nDimension;
            THFloatStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THFloatTensor; -}
#starttype struct THFloatTensor
#field size , Ptr CLong
#field stride , Ptr CLong
#field nDimension , CInt
#field storage , Ptr <struct THFloatStorage>
#field storageOffset , CLong
#field refcount , CInt
#field flag , CChar
#stoptype

{- typedef struct THDoubleStorage {
            double * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THDoubleStorage * view;
        } THDoubleStorage; -}
#starttype struct THDoubleStorage
#field data , Ptr CDouble
#field size , CLong
#field refcount , CInt
#field flag , CChar
#field allocator , Ptr <struct THAllocator>
#field allocatorContext , Ptr ()
#field view , Ptr <struct THDoubleStorage>
#stoptype

{- typedef struct THDoubleTensor {
            long * size;
            long * stride;
            int nDimension;
            THDoubleStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THDoubleTensor; -}
#starttype struct THDoubleTensor
#field size , Ptr CLong
#field stride , Ptr CLong
#field nDimension , CInt
#field storage , Ptr <struct THDoubleStorage>
#field storageOffset , CLong
#field refcount , CInt
#field flag , CChar
#stoptype

{- typedef struct THIntStorage {
            int * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THIntStorage * view;
        } THIntStorage; -}
#starttype struct THIntStorage
#field data , Ptr CInt
#field size , CLong
#field refcount , CInt
#field flag , CChar
#field allocator , Ptr <struct THAllocator>
#field allocatorContext , Ptr ()
#field view , Ptr <struct THIntStorage>
#stoptype

{- typedef struct THIntTensor {
            long * size;
            long * stride;
            int nDimension;
            THIntStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THIntTensor; -}
#starttype struct THIntTensor
#field size , Ptr CLong
#field stride , Ptr CLong
#field nDimension , CInt
#field storage , Ptr <struct THIntStorage>
#field storageOffset , CLong
#field refcount , CInt
#field flag , CChar
#stoptype

{- typedef struct THCharStorage {
            char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THCharStorage * view;
        } THCharStorage; -}
#starttype struct THCharStorage
#field data , CString
#field size , CLong
#field refcount , CInt
#field flag , CChar
#field allocator , Ptr <struct THAllocator>
#field allocatorContext , Ptr ()
#field view , Ptr <struct THCharStorage>
#stoptype

{- typedef struct THCharTensor {
            long * size;
            long * stride;
            int nDimension;
            THCharStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCharTensor; -}
#starttype struct THCharTensor
#field size , Ptr CLong
#field stride , Ptr CLong
#field nDimension , CInt
#field storage , Ptr <struct THCharStorage>
#field storageOffset , CLong
#field refcount , CInt
#field flag , CChar
#stoptype

{- typedef struct THByteStorage {
            unsigned char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THByteStorage * view;
        } THByteStorage; -}
#starttype struct THByteStorage
#field data , Ptr CUChar
#field size , CLong
#field refcount , CInt
#field flag , CChar
#field allocator , Ptr <struct THAllocator>
#field allocatorContext , Ptr ()
#field view , Ptr <struct THByteStorage>
#stoptype

{- typedef struct THByteTensor {
            long * size;
            long * stride;
            int nDimension;
            THByteStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THByteTensor; -}
#starttype struct THByteTensor
#field size , Ptr CLong
#field stride , Ptr CLong
#field nDimension , CInt
#field storage , Ptr <struct THByteStorage>
#field storageOffset , CLong
#field refcount , CInt
#field flag , CChar
#stoptype

