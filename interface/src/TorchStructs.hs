{-# LINE 1 "TorchStructs.hsc" #-}
{-# OPTIONS_GHC -fno-warn-unused-imports #-}
{-# LINE 2 "TorchStructs.hsc" #-}

{-# LINE 3 "TorchStructs.hsc" #-}

{-# LINE 4 "TorchStructs.hsc" #-}
module TorchStructs where
import Foreign.Ptr
import Foreign.Ptr (Ptr,FunPtr,plusPtr)
import Foreign.Ptr (wordPtrToPtr,castPtrToFunPtr)
import Foreign.Storable
import Foreign.C.Types
import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray,pokeArray)
import Data.Int
import Data.Word

{-# LINE 7 "TorchStructs.hsc" #-}

{- typedef struct THAllocator {
            void * (* malloc)(void *, ptrdiff_t);
            void * (* realloc)(void *, void *, ptrdiff_t);
            void (* free)(void *, void *);
        } THAllocator; -}

{-# LINE 14 "TorchStructs.hsc" #-}

{-# LINE 15 "TorchStructs.hsc" #-}

{-# LINE 16 "TorchStructs.hsc" #-}

{-# LINE 17 "TorchStructs.hsc" #-}
data C'THAllocator = C'THAllocator{
  c'THAllocator'malloc :: FunPtr (Ptr () -> CLong -> Ptr ()),
  c'THAllocator'realloc :: FunPtr (Ptr () -> Ptr () -> CLong -> Ptr ()),
  c'THAllocator'free :: FunPtr (Ptr () -> Ptr () -> IO ())
} deriving (Eq,Show)
p'THAllocator'malloc p = plusPtr p 0
p'THAllocator'malloc :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> CLong -> Ptr ()))
p'THAllocator'realloc p = plusPtr p 8
p'THAllocator'realloc :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> Ptr () -> CLong -> Ptr ()))
p'THAllocator'free p = plusPtr p 16
p'THAllocator'free :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> Ptr () -> IO ()))
instance Storable C'THAllocator where
  sizeOf _ = 24
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    return $ C'THAllocator v0 v1 v2
  poke _p (C'THAllocator v0 v1 v2) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    return ()

{-# LINE 18 "TorchStructs.hsc" #-}

{- typedef struct THDoubleStorage {
            double * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THDoubleStorage * view;
        } THDoubleStorage; -}

{-# LINE 29 "TorchStructs.hsc" #-}

{-# LINE 30 "TorchStructs.hsc" #-}

{-# LINE 31 "TorchStructs.hsc" #-}

{-# LINE 32 "TorchStructs.hsc" #-}

{-# LINE 33 "TorchStructs.hsc" #-}

{-# LINE 34 "TorchStructs.hsc" #-}

{-# LINE 35 "TorchStructs.hsc" #-}

{-# LINE 36 "TorchStructs.hsc" #-}
data C'THDoubleStorage = C'THDoubleStorage{
  c'THDoubleStorage'data :: Ptr CDouble,
  c'THDoubleStorage'size :: CLong,
  c'THDoubleStorage'refcount :: CInt,
  c'THDoubleStorage'flag :: CChar,
  c'THDoubleStorage'allocator :: Ptr C'THAllocator,
  c'THDoubleStorage'allocatorContext :: Ptr (),
  c'THDoubleStorage'view :: Ptr C'THDoubleStorage
} deriving (Eq,Show)
p'THDoubleStorage'data p = plusPtr p 0
p'THDoubleStorage'data :: Ptr (C'THDoubleStorage) -> Ptr (Ptr CDouble)
p'THDoubleStorage'size p = plusPtr p 8
p'THDoubleStorage'size :: Ptr (C'THDoubleStorage) -> Ptr (CLong)
p'THDoubleStorage'refcount p = plusPtr p 16
p'THDoubleStorage'refcount :: Ptr (C'THDoubleStorage) -> Ptr (CInt)
p'THDoubleStorage'flag p = plusPtr p 20
p'THDoubleStorage'flag :: Ptr (C'THDoubleStorage) -> Ptr (CChar)
p'THDoubleStorage'allocator p = plusPtr p 24
p'THDoubleStorage'allocator :: Ptr (C'THDoubleStorage) -> Ptr (Ptr C'THAllocator)
p'THDoubleStorage'allocatorContext p = plusPtr p 32
p'THDoubleStorage'allocatorContext :: Ptr (C'THDoubleStorage) -> Ptr (Ptr ())
p'THDoubleStorage'view p = plusPtr p 40
p'THDoubleStorage'view :: Ptr (C'THDoubleStorage) -> Ptr (Ptr C'THDoubleStorage)
instance Storable C'THDoubleStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THDoubleStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THDoubleStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 37 "TorchStructs.hsc" #-}

{- typedef struct THDoubleTensor {
            long * size;
            long * stride;
            int nDimension;
            THDoubleStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THDoubleTensor; -}

{-# LINE 48 "TorchStructs.hsc" #-}

{-# LINE 49 "TorchStructs.hsc" #-}

{-# LINE 50 "TorchStructs.hsc" #-}

{-# LINE 51 "TorchStructs.hsc" #-}

{-# LINE 52 "TorchStructs.hsc" #-}

{-# LINE 53 "TorchStructs.hsc" #-}

{-# LINE 54 "TorchStructs.hsc" #-}

{-# LINE 55 "TorchStructs.hsc" #-}
data C'THDoubleTensor = C'THDoubleTensor{
  c'THDoubleTensor'size :: Ptr CLong,
  c'THDoubleTensor'stride :: Ptr CLong,
  c'THDoubleTensor'nDimension :: CInt,
  c'THDoubleTensor'storage :: Ptr C'THDoubleStorage,
  c'THDoubleTensor'storageOffset :: CLong,
  c'THDoubleTensor'refcount :: CInt,
  c'THDoubleTensor'flag :: CChar
} deriving (Eq,Show)
p'THDoubleTensor'size p = plusPtr p 0
p'THDoubleTensor'size :: Ptr (C'THDoubleTensor) -> Ptr (Ptr CLong)
p'THDoubleTensor'stride p = plusPtr p 8
p'THDoubleTensor'stride :: Ptr (C'THDoubleTensor) -> Ptr (Ptr CLong)
p'THDoubleTensor'nDimension p = plusPtr p 16
p'THDoubleTensor'nDimension :: Ptr (C'THDoubleTensor) -> Ptr (CInt)
p'THDoubleTensor'storage p = plusPtr p 24
p'THDoubleTensor'storage :: Ptr (C'THDoubleTensor) -> Ptr (Ptr C'THDoubleStorage)
p'THDoubleTensor'storageOffset p = plusPtr p 32
p'THDoubleTensor'storageOffset :: Ptr (C'THDoubleTensor) -> Ptr (CLong)
p'THDoubleTensor'refcount p = plusPtr p 40
p'THDoubleTensor'refcount :: Ptr (C'THDoubleTensor) -> Ptr (CInt)
p'THDoubleTensor'flag p = plusPtr p 44
p'THDoubleTensor'flag :: Ptr (C'THDoubleTensor) -> Ptr (CChar)
instance Storable C'THDoubleTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THDoubleTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THDoubleTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 56 "TorchStructs.hsc" #-}

