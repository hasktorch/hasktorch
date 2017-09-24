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

