{-# OPTIONS_GHC -fno-warn-unused-imports #-}


module Torch.Types.THC.Structs where
import Torch.Types.THC.Internal
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


{- typedef struct THCStream {
            cudaStream_t stream; int device; int refcount;
        } THCStream; -}




data C'THCStream = C'THCStream{
  c'THCStream'stream :: C'cudaStream_t,
  c'THCStream'device :: CInt,
  c'THCStream'refcount :: CInt
} deriving (Eq,Show)
p'THCStream'stream p = plusPtr p 0
p'THCStream'stream :: Ptr (C'THCStream) -> Ptr (C'cudaStream_t)
p'THCStream'device p = plusPtr p 0
p'THCStream'device :: Ptr (C'THCStream) -> Ptr (CInt)
p'THCStream'refcount p = plusPtr p 4
p'THCStream'refcount :: Ptr (C'THCStream) -> Ptr (CInt)
instance Storable C'THCStream where
  sizeOf _ = 8
  alignment _ = 4
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 0
    v2 <- peekByteOff _p 4
    return $ C'THCStream v0 v1 v2
  poke _p (C'THCStream v0 v1 v2) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 0 v1
    pokeByteOff _p 4 v2
    return ()


{- typedef struct _Generator {
            struct curandStateMtgp32 * gen_states;
            struct mtgp32_kernel_params * kernel_params;
            int initf;
            unsigned long long initial_seed;
        } Generator; -}





data C'_Generator = C'_Generator{
  c'_Generator'gen_states :: Ptr C'curandStateMtgp32,
  c'_Generator'kernel_params :: Ptr C'mtgp32_kernel_params,
  c'_Generator'initf :: CInt,
  c'_Generator'initial_seed :: CULong
} deriving (Eq,Show)
p'_Generator'gen_states p = plusPtr p 0
p'_Generator'gen_states :: Ptr (C'_Generator) -> Ptr (Ptr C'curandStateMtgp32)
p'_Generator'kernel_params p = plusPtr p 8
p'_Generator'kernel_params :: Ptr (C'_Generator) -> Ptr (Ptr C'mtgp32_kernel_params)
p'_Generator'initf p = plusPtr p 16
p'_Generator'initf :: Ptr (C'_Generator) -> Ptr (CInt)
p'_Generator'initial_seed p = plusPtr p 24
p'_Generator'initial_seed :: Ptr (C'_Generator) -> Ptr (CULong)
instance Storable C'_Generator where
  sizeOf _ = 32
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    return $ C'_Generator v0 v1 v2 v3
  poke _p (C'_Generator v0 v1 v2 v3) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    return ()


{- typedef struct THCRNGState {
            Generator * gen; int num_devices;
        } THCRNGState; -}



data C'THCRNGState = C'THCRNGState{
  c'THCRNGState'gen :: Ptr C'_Generator,
  c'THCRNGState'num_devices :: CInt
} deriving (Eq,Show)
p'THCRNGState'gen p = plusPtr p 0
p'THCRNGState'gen :: Ptr (C'THCRNGState) -> Ptr (Ptr C'_Generator)
p'THCRNGState'num_devices p = plusPtr p 8
p'THCRNGState'num_devices :: Ptr (C'THCRNGState) -> Ptr (CInt)
instance Storable C'THCRNGState where
  sizeOf _ = 16
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    return $ C'THCRNGState v0 v1
  poke _p (C'THCRNGState v0 v1) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    return ()


{- typedef struct THCDescBuff {
            char str[64];
        } THCDescBuff; -}


data C'THCDescBuff = C'THCDescBuff{
  c'THCDescBuff'str :: [CChar]
} deriving (Eq,Show)
p'THCDescBuff'str p = plusPtr p 0
p'THCDescBuff'str :: Ptr (C'THCDescBuff) -> Ptr (CChar)
instance Storable C'THCDescBuff where
  sizeOf _ = 64
  alignment _ = 1
  peek _p = do
    v0 <- let s0 = div 64 $ sizeOf $ (undefined :: CChar) in peekArray s0 (plusPtr _p 0)
    return $ C'THCDescBuff v0
  poke _p (C'THCDescBuff v0) = do
    let s0 = div 64 $ sizeOf $ (undefined :: CChar)
    pokeArray (plusPtr _p 0) (take s0 v0)
    return ()


{- typedef struct _THCDeviceAllocator {
            cudaError_t (* malloc)(void *, void * *, size_t, cudaStream_t);
            cudaError_t (* realloc)(void *,
                                    void * *,
                                    size_t,
                                    size_t,
                                    cudaStream_t);
            cudaError_t (* free)(void *, void *);
            cudaError_t (* emptyCache)(void *);
            cudaError_t (* cacheInfo)(void *, int, size_t *, size_t *);
            void * state;
        } THCDeviceAllocator; -}







data C'_THCDeviceAllocator = C'_THCDeviceAllocator{
  c'_THCDeviceAllocator'malloc :: FunPtr (Ptr () -> Ptr (Ptr ()) -> CSize -> C'cudaStream_t -> C'cudaError_t),
  c'_THCDeviceAllocator'realloc :: FunPtr (Ptr () -> Ptr (Ptr ()) -> CSize -> CSize -> C'cudaStream_t -> C'cudaError_t),
  c'_THCDeviceAllocator'free :: FunPtr (Ptr () -> Ptr () -> C'cudaError_t),
  c'_THCDeviceAllocator'emptyCache :: FunPtr (Ptr () -> C'cudaError_t),
  c'_THCDeviceAllocator'cacheInfo :: FunPtr (Ptr () -> CInt -> Ptr CSize -> Ptr CSize -> C'cudaError_t),
  c'_THCDeviceAllocator'state :: Ptr ()
} deriving (Eq,Show)
p'_THCDeviceAllocator'malloc p = plusPtr p 0
p'_THCDeviceAllocator'malloc :: Ptr (C'_THCDeviceAllocator) -> Ptr (FunPtr (Ptr () -> Ptr (Ptr ()) -> CSize -> C'cudaStream_t -> C'cudaError_t))
p'_THCDeviceAllocator'realloc p = plusPtr p 8
p'_THCDeviceAllocator'realloc :: Ptr (C'_THCDeviceAllocator) -> Ptr (FunPtr (Ptr () -> Ptr (Ptr ()) -> CSize -> CSize -> C'cudaStream_t -> C'cudaError_t))
p'_THCDeviceAllocator'free p = plusPtr p 16
p'_THCDeviceAllocator'free :: Ptr (C'_THCDeviceAllocator) -> Ptr (FunPtr (Ptr () -> Ptr () -> C'cudaError_t))
p'_THCDeviceAllocator'emptyCache p = plusPtr p 24
p'_THCDeviceAllocator'emptyCache :: Ptr (C'_THCDeviceAllocator) -> Ptr (FunPtr (Ptr () -> C'cudaError_t))
p'_THCDeviceAllocator'cacheInfo p = plusPtr p 32
p'_THCDeviceAllocator'cacheInfo :: Ptr (C'_THCDeviceAllocator) -> Ptr (FunPtr (Ptr () -> CInt -> Ptr CSize -> Ptr CSize -> C'cudaError_t))
p'_THCDeviceAllocator'state p = plusPtr p 40
p'_THCDeviceAllocator'state :: Ptr (C'_THCDeviceAllocator) -> Ptr (Ptr ())
instance Storable C'_THCDeviceAllocator where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    return $ C'_THCDeviceAllocator v0 v1 v2 v3 v4 v5
  poke _p (C'_THCDeviceAllocator v0 v1 v2 v3 v4 v5) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    return ()


{- typedef struct _THCCudaResourcesPerDevice {
            THCStream * * streams;
            int numBlasHandles;
            int numSparseHandles;
            cublasHandle_t * blasHandles;
            cusparseHandle_t * sparseHandles;
            size_t scratchSpacePerStream;
            void * * devScratchSpacePerStream;
        } THCCudaResourcesPerDevice; -}








data C'_THCCudaResourcesPerDevice = C'_THCCudaResourcesPerDevice{
  c'_THCCudaResourcesPerDevice'streams :: Ptr (Ptr C'THCStream),
  c'_THCCudaResourcesPerDevice'numBlasHandles :: CInt,
  c'_THCCudaResourcesPerDevice'numSparseHandles :: CInt,
  c'_THCCudaResourcesPerDevice'blasHandles :: Ptr C'cublasHandle_t,
  c'_THCCudaResourcesPerDevice'sparseHandles :: Ptr C'cusparseHandle_t,
  c'_THCCudaResourcesPerDevice'scratchSpacePerStream :: CSize,
  c'_THCCudaResourcesPerDevice'devScratchSpacePerStream :: Ptr (Ptr ())
} deriving (Eq,Show)
p'_THCCudaResourcesPerDevice'streams p = plusPtr p 0
p'_THCCudaResourcesPerDevice'streams :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (Ptr (Ptr C'THCStream))
p'_THCCudaResourcesPerDevice'numBlasHandles p = plusPtr p 8
p'_THCCudaResourcesPerDevice'numBlasHandles :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (CInt)
p'_THCCudaResourcesPerDevice'numSparseHandles p = plusPtr p 12
p'_THCCudaResourcesPerDevice'numSparseHandles :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (CInt)
p'_THCCudaResourcesPerDevice'blasHandles p = plusPtr p 16
p'_THCCudaResourcesPerDevice'blasHandles :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (Ptr C'cublasHandle_t)
p'_THCCudaResourcesPerDevice'sparseHandles p = plusPtr p 24
p'_THCCudaResourcesPerDevice'sparseHandles :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (Ptr C'cusparseHandle_t)
p'_THCCudaResourcesPerDevice'scratchSpacePerStream p = plusPtr p 32
p'_THCCudaResourcesPerDevice'scratchSpacePerStream :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (CSize)
p'_THCCudaResourcesPerDevice'devScratchSpacePerStream p = plusPtr p 40
p'_THCCudaResourcesPerDevice'devScratchSpacePerStream :: Ptr (C'_THCCudaResourcesPerDevice) -> Ptr (Ptr (Ptr ()))
instance Storable C'_THCCudaResourcesPerDevice where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 12
    v3 <- peekByteOff _p 16
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'_THCCudaResourcesPerDevice v0 v1 v2 v3 v4 v5 v6
  poke _p (C'_THCCudaResourcesPerDevice v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 12 v2
    pokeByteOff _p 16 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()


{- typedef struct THCState {
            struct THCRNGState * rngState;
            struct cudaDeviceProp * deviceProperties;
            THCCudaResourcesPerDevice * resourcesPerDevice;
            int numDevices;
            int numUserStreams;
            int numUserBlasHandles;
            int numUserSparseHandles;
            THAllocator * cudaHostAllocator;
            THAllocator * cudaUVAAllocator;
            THCDeviceAllocator * cudaDeviceAllocator;
            int * * p2pAccessEnabled;
            int p2pKernelAccessEnabled;
            void (* cutorchGCFunction)(void * data);
            void * cutorchGCData;
            ptrdiff_t heapSoftmax;
            ptrdiff_t heapDelta;
        } THCState; -}

















data C'THCState = C'THCState{
  c'THCState'rngState :: Ptr C'THCRNGState,
  c'THCState'deviceProperties :: Ptr C'cudaDeviceProp,
  c'THCState'resourcesPerDevice :: Ptr C'_THCCudaResourcesPerDevice,
  c'THCState'numDevices :: CInt,
  c'THCState'numUserStreams :: CInt,
  c'THCState'numUserBlasHandles :: CInt,
  c'THCState'numUserSparseHandles :: CInt,
  c'THCState'cudaHostAllocator :: Ptr C'THAllocator,
  c'THCState'cudaUVAAllocator :: Ptr C'THAllocator,
  c'THCState'cudaDeviceAllocator :: Ptr C'_THCDeviceAllocator,
  c'THCState'p2pAccessEnabled :: Ptr (Ptr CInt),
  c'THCState'p2pKernelAccessEnabled :: CInt,
  c'THCState'cutorchGCFunction :: FunPtr (Ptr () -> IO ()),
  c'THCState'cutorchGCData :: Ptr (),
  c'THCState'heapSoftmax :: CLong,
  c'THCState'heapDelta :: CLong
} deriving (Eq,Show)
p'THCState'rngState p = plusPtr p 0
p'THCState'rngState :: Ptr (C'THCState) -> Ptr (Ptr C'THCRNGState)
p'THCState'deviceProperties p = plusPtr p 8
p'THCState'deviceProperties :: Ptr (C'THCState) -> Ptr (Ptr C'cudaDeviceProp)
p'THCState'resourcesPerDevice p = plusPtr p 16
p'THCState'resourcesPerDevice :: Ptr (C'THCState) -> Ptr (Ptr C'_THCCudaResourcesPerDevice)
p'THCState'numDevices p = plusPtr p 24
p'THCState'numDevices :: Ptr (C'THCState) -> Ptr (CInt)
p'THCState'numUserStreams p = plusPtr p 28
p'THCState'numUserStreams :: Ptr (C'THCState) -> Ptr (CInt)
p'THCState'numUserBlasHandles p = plusPtr p 32
p'THCState'numUserBlasHandles :: Ptr (C'THCState) -> Ptr (CInt)
p'THCState'numUserSparseHandles p = plusPtr p 36
p'THCState'numUserSparseHandles :: Ptr (C'THCState) -> Ptr (CInt)
p'THCState'cudaHostAllocator p = plusPtr p 40
p'THCState'cudaHostAllocator :: Ptr (C'THCState) -> Ptr (Ptr C'THAllocator)
p'THCState'cudaUVAAllocator p = plusPtr p 48
p'THCState'cudaUVAAllocator :: Ptr (C'THCState) -> Ptr (Ptr C'THAllocator)
p'THCState'cudaDeviceAllocator p = plusPtr p 56
p'THCState'cudaDeviceAllocator :: Ptr (C'THCState) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCState'p2pAccessEnabled p = plusPtr p 64
p'THCState'p2pAccessEnabled :: Ptr (C'THCState) -> Ptr (Ptr (Ptr CInt))
p'THCState'p2pKernelAccessEnabled p = plusPtr p 72
p'THCState'p2pKernelAccessEnabled :: Ptr (C'THCState) -> Ptr (CInt)
p'THCState'cutorchGCFunction p = plusPtr p 80
p'THCState'cutorchGCFunction :: Ptr (C'THCState) -> Ptr (FunPtr (Ptr () -> IO ()))
p'THCState'cutorchGCData p = plusPtr p 88
p'THCState'cutorchGCData :: Ptr (C'THCState) -> Ptr (Ptr ())
p'THCState'heapSoftmax p = plusPtr p 96
p'THCState'heapSoftmax :: Ptr (C'THCState) -> Ptr (CLong)
p'THCState'heapDelta p = plusPtr p 104
p'THCState'heapDelta :: Ptr (C'THCState) -> Ptr (CLong)
instance Storable C'THCState where
  sizeOf _ = 112
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 28
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 36
    v7 <- peekByteOff _p 40
    v8 <- peekByteOff _p 48
    v9 <- peekByteOff _p 56
    v10 <- peekByteOff _p 64
    v11 <- peekByteOff _p 72
    v12 <- peekByteOff _p 80
    v13 <- peekByteOff _p 88
    v14 <- peekByteOff _p 96
    v15 <- peekByteOff _p 104
    return $ C'THCState v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15
  poke _p (C'THCState v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 28 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 36 v6
    pokeByteOff _p 40 v7
    pokeByteOff _p 48 v8
    pokeByteOff _p 56 v9
    pokeByteOff _p 64 v10
    pokeByteOff _p 72 v11
    pokeByteOff _p 80 v12
    pokeByteOff _p 88 v13
    pokeByteOff _p 96 v14
    pokeByteOff _p 104 v15
    return ()


{- typedef struct THCByteStorage {
            unsigned char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCByteStorage * view;
            int device;
        } THCByteStorage; -}









data C'THCByteStorage = C'THCByteStorage{
  c'THCByteStorage'data :: Ptr CUChar,
  c'THCByteStorage'size :: CLong,
  c'THCByteStorage'refcount :: CInt,
  c'THCByteStorage'flag :: CChar,
  c'THCByteStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCByteStorage'allocatorContext :: Ptr (),
  c'THCByteStorage'view :: Ptr C'THCByteStorage,
  c'THCByteStorage'device :: CInt
} deriving (Eq,Show)
p'THCByteStorage'data p = plusPtr p 0
p'THCByteStorage'data :: Ptr (C'THCByteStorage) -> Ptr (Ptr CUChar)
p'THCByteStorage'size p = plusPtr p 8
p'THCByteStorage'size :: Ptr (C'THCByteStorage) -> Ptr (CLong)
p'THCByteStorage'refcount p = plusPtr p 16
p'THCByteStorage'refcount :: Ptr (C'THCByteStorage) -> Ptr (CInt)
p'THCByteStorage'flag p = plusPtr p 20
p'THCByteStorage'flag :: Ptr (C'THCByteStorage) -> Ptr (CChar)
p'THCByteStorage'allocator p = plusPtr p 24
p'THCByteStorage'allocator :: Ptr (C'THCByteStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCByteStorage'allocatorContext p = plusPtr p 32
p'THCByteStorage'allocatorContext :: Ptr (C'THCByteStorage) -> Ptr (Ptr ())
p'THCByteStorage'view p = plusPtr p 40
p'THCByteStorage'view :: Ptr (C'THCByteStorage) -> Ptr (Ptr C'THCByteStorage)
p'THCByteStorage'device p = plusPtr p 48
p'THCByteStorage'device :: Ptr (C'THCByteStorage) -> Ptr (CInt)
instance Storable C'THCByteStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCByteStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCByteStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCCharStorage {
            char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCCharStorage * view;
            int device;
        } THCCharStorage; -}









data C'THCCharStorage = C'THCCharStorage{
  c'THCCharStorage'data :: CString,
  c'THCCharStorage'size :: CLong,
  c'THCCharStorage'refcount :: CInt,
  c'THCCharStorage'flag :: CChar,
  c'THCCharStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCCharStorage'allocatorContext :: Ptr (),
  c'THCCharStorage'view :: Ptr C'THCCharStorage,
  c'THCCharStorage'device :: CInt
} deriving (Eq,Show)
p'THCCharStorage'data p = plusPtr p 0
p'THCCharStorage'data :: Ptr (C'THCCharStorage) -> Ptr (CString)
p'THCCharStorage'size p = plusPtr p 8
p'THCCharStorage'size :: Ptr (C'THCCharStorage) -> Ptr (CLong)
p'THCCharStorage'refcount p = plusPtr p 16
p'THCCharStorage'refcount :: Ptr (C'THCCharStorage) -> Ptr (CInt)
p'THCCharStorage'flag p = plusPtr p 20
p'THCCharStorage'flag :: Ptr (C'THCCharStorage) -> Ptr (CChar)
p'THCCharStorage'allocator p = plusPtr p 24
p'THCCharStorage'allocator :: Ptr (C'THCCharStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCCharStorage'allocatorContext p = plusPtr p 32
p'THCCharStorage'allocatorContext :: Ptr (C'THCCharStorage) -> Ptr (Ptr ())
p'THCCharStorage'view p = plusPtr p 40
p'THCCharStorage'view :: Ptr (C'THCCharStorage) -> Ptr (Ptr C'THCCharStorage)
p'THCCharStorage'device p = plusPtr p 48
p'THCCharStorage'device :: Ptr (C'THCCharStorage) -> Ptr (CInt)
instance Storable C'THCCharStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCCharStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCCharStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCDoubleStorage {
            double * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCDoubleStorage * view;
            int device;
        } THCDoubleStorage; -}









data C'THCDoubleStorage = C'THCDoubleStorage{
  c'THCDoubleStorage'data :: Ptr CDouble,
  c'THCDoubleStorage'size :: CLong,
  c'THCDoubleStorage'refcount :: CInt,
  c'THCDoubleStorage'flag :: CChar,
  c'THCDoubleStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCDoubleStorage'allocatorContext :: Ptr (),
  c'THCDoubleStorage'view :: Ptr C'THCDoubleStorage,
  c'THCDoubleStorage'device :: CInt
} deriving (Eq,Show)
p'THCDoubleStorage'data p = plusPtr p 0
p'THCDoubleStorage'data :: Ptr (C'THCDoubleStorage) -> Ptr (Ptr CDouble)
p'THCDoubleStorage'size p = plusPtr p 8
p'THCDoubleStorage'size :: Ptr (C'THCDoubleStorage) -> Ptr (CLong)
p'THCDoubleStorage'refcount p = plusPtr p 16
p'THCDoubleStorage'refcount :: Ptr (C'THCDoubleStorage) -> Ptr (CInt)
p'THCDoubleStorage'flag p = plusPtr p 20
p'THCDoubleStorage'flag :: Ptr (C'THCDoubleStorage) -> Ptr (CChar)
p'THCDoubleStorage'allocator p = plusPtr p 24
p'THCDoubleStorage'allocator :: Ptr (C'THCDoubleStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCDoubleStorage'allocatorContext p = plusPtr p 32
p'THCDoubleStorage'allocatorContext :: Ptr (C'THCDoubleStorage) -> Ptr (Ptr ())
p'THCDoubleStorage'view p = plusPtr p 40
p'THCDoubleStorage'view :: Ptr (C'THCDoubleStorage) -> Ptr (Ptr C'THCDoubleStorage)
p'THCDoubleStorage'device p = plusPtr p 48
p'THCDoubleStorage'device :: Ptr (C'THCDoubleStorage) -> Ptr (CInt)
instance Storable C'THCDoubleStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCDoubleStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCDoubleStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCFloatStorage {
            float * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCFloatStorage * view;
            int device;
        } THCFloatStorage; -}









data C'THCFloatStorage = C'THCFloatStorage{
  c'THCFloatStorage'data :: Ptr CFloat,
  c'THCFloatStorage'size :: CLong,
  c'THCFloatStorage'refcount :: CInt,
  c'THCFloatStorage'flag :: CChar,
  c'THCFloatStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCFloatStorage'allocatorContext :: Ptr (),
  c'THCFloatStorage'view :: Ptr C'THCFloatStorage,
  c'THCFloatStorage'device :: CInt
} deriving (Eq,Show)
p'THCFloatStorage'data p = plusPtr p 0
p'THCFloatStorage'data :: Ptr (C'THCFloatStorage) -> Ptr (Ptr CFloat)
p'THCFloatStorage'size p = plusPtr p 8
p'THCFloatStorage'size :: Ptr (C'THCFloatStorage) -> Ptr (CLong)
p'THCFloatStorage'refcount p = plusPtr p 16
p'THCFloatStorage'refcount :: Ptr (C'THCFloatStorage) -> Ptr (CInt)
p'THCFloatStorage'flag p = plusPtr p 20
p'THCFloatStorage'flag :: Ptr (C'THCFloatStorage) -> Ptr (CChar)
p'THCFloatStorage'allocator p = plusPtr p 24
p'THCFloatStorage'allocator :: Ptr (C'THCFloatStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCFloatStorage'allocatorContext p = plusPtr p 32
p'THCFloatStorage'allocatorContext :: Ptr (C'THCFloatStorage) -> Ptr (Ptr ())
p'THCFloatStorage'view p = plusPtr p 40
p'THCFloatStorage'view :: Ptr (C'THCFloatStorage) -> Ptr (Ptr C'THCFloatStorage)
p'THCFloatStorage'device p = plusPtr p 48
p'THCFloatStorage'device :: Ptr (C'THCFloatStorage) -> Ptr (CInt)
instance Storable C'THCFloatStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCFloatStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCFloatStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCIntStorage {
            int * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCIntStorage * view;
            int device;
        } THCIntStorage; -}









data C'THCIntStorage = C'THCIntStorage{
  c'THCIntStorage'data :: Ptr CInt,
  c'THCIntStorage'size :: CLong,
  c'THCIntStorage'refcount :: CInt,
  c'THCIntStorage'flag :: CChar,
  c'THCIntStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCIntStorage'allocatorContext :: Ptr (),
  c'THCIntStorage'view :: Ptr C'THCIntStorage,
  c'THCIntStorage'device :: CInt
} deriving (Eq,Show)
p'THCIntStorage'data p = plusPtr p 0
p'THCIntStorage'data :: Ptr (C'THCIntStorage) -> Ptr (Ptr CInt)
p'THCIntStorage'size p = plusPtr p 8
p'THCIntStorage'size :: Ptr (C'THCIntStorage) -> Ptr (CLong)
p'THCIntStorage'refcount p = plusPtr p 16
p'THCIntStorage'refcount :: Ptr (C'THCIntStorage) -> Ptr (CInt)
p'THCIntStorage'flag p = plusPtr p 20
p'THCIntStorage'flag :: Ptr (C'THCIntStorage) -> Ptr (CChar)
p'THCIntStorage'allocator p = plusPtr p 24
p'THCIntStorage'allocator :: Ptr (C'THCIntStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCIntStorage'allocatorContext p = plusPtr p 32
p'THCIntStorage'allocatorContext :: Ptr (C'THCIntStorage) -> Ptr (Ptr ())
p'THCIntStorage'view p = plusPtr p 40
p'THCIntStorage'view :: Ptr (C'THCIntStorage) -> Ptr (Ptr C'THCIntStorage)
p'THCIntStorage'device p = plusPtr p 48
p'THCIntStorage'device :: Ptr (C'THCIntStorage) -> Ptr (CInt)
instance Storable C'THCIntStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCIntStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCIntStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCShortStorage {
            short * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCShortStorage * view;
            int device;
        } THCShortStorage; -}









data C'THCShortStorage = C'THCShortStorage{
  c'THCShortStorage'data :: Ptr CShort,
  c'THCShortStorage'size :: CLong,
  c'THCShortStorage'refcount :: CInt,
  c'THCShortStorage'flag :: CChar,
  c'THCShortStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCShortStorage'allocatorContext :: Ptr (),
  c'THCShortStorage'view :: Ptr C'THCShortStorage,
  c'THCShortStorage'device :: CInt
} deriving (Eq,Show)
p'THCShortStorage'data p = plusPtr p 0
p'THCShortStorage'data :: Ptr (C'THCShortStorage) -> Ptr (Ptr CShort)
p'THCShortStorage'size p = plusPtr p 8
p'THCShortStorage'size :: Ptr (C'THCShortStorage) -> Ptr (CLong)
p'THCShortStorage'refcount p = plusPtr p 16
p'THCShortStorage'refcount :: Ptr (C'THCShortStorage) -> Ptr (CInt)
p'THCShortStorage'flag p = plusPtr p 20
p'THCShortStorage'flag :: Ptr (C'THCShortStorage) -> Ptr (CChar)
p'THCShortStorage'allocator p = plusPtr p 24
p'THCShortStorage'allocator :: Ptr (C'THCShortStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCShortStorage'allocatorContext p = plusPtr p 32
p'THCShortStorage'allocatorContext :: Ptr (C'THCShortStorage) -> Ptr (Ptr ())
p'THCShortStorage'view p = plusPtr p 40
p'THCShortStorage'view :: Ptr (C'THCShortStorage) -> Ptr (Ptr C'THCShortStorage)
p'THCShortStorage'device p = plusPtr p 48
p'THCShortStorage'device :: Ptr (C'THCShortStorage) -> Ptr (CInt)
instance Storable C'THCShortStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCShortStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCShortStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCLongStorage {
            long * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THCDeviceAllocator * allocator;
            void * allocatorContext;
            struct THCLongStorage * view;
            int device;
        } THCLongStorage; -}









data C'THCLongStorage = C'THCLongStorage{
  c'THCLongStorage'data :: Ptr CLong,
  c'THCLongStorage'size :: CLong,
  c'THCLongStorage'refcount :: CInt,
  c'THCLongStorage'flag :: CChar,
  c'THCLongStorage'allocator :: Ptr C'_THCDeviceAllocator,
  c'THCLongStorage'allocatorContext :: Ptr (),
  c'THCLongStorage'view :: Ptr C'THCLongStorage,
  c'THCLongStorage'device :: CInt
} deriving (Eq,Show)
p'THCLongStorage'data p = plusPtr p 0
p'THCLongStorage'data :: Ptr (C'THCLongStorage) -> Ptr (Ptr CLong)
p'THCLongStorage'size p = plusPtr p 8
p'THCLongStorage'size :: Ptr (C'THCLongStorage) -> Ptr (CLong)
p'THCLongStorage'refcount p = plusPtr p 16
p'THCLongStorage'refcount :: Ptr (C'THCLongStorage) -> Ptr (CInt)
p'THCLongStorage'flag p = plusPtr p 20
p'THCLongStorage'flag :: Ptr (C'THCLongStorage) -> Ptr (CChar)
p'THCLongStorage'allocator p = plusPtr p 24
p'THCLongStorage'allocator :: Ptr (C'THCLongStorage) -> Ptr (Ptr C'_THCDeviceAllocator)
p'THCLongStorage'allocatorContext p = plusPtr p 32
p'THCLongStorage'allocatorContext :: Ptr (C'THCLongStorage) -> Ptr (Ptr ())
p'THCLongStorage'view p = plusPtr p 40
p'THCLongStorage'view :: Ptr (C'THCLongStorage) -> Ptr (Ptr C'THCLongStorage)
p'THCLongStorage'device p = plusPtr p 48
p'THCLongStorage'device :: Ptr (C'THCLongStorage) -> Ptr (CInt)
instance Storable C'THCLongStorage where
  sizeOf _ = 56
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    v7 <- peekByteOff _p 48
    return $ C'THCLongStorage v0 v1 v2 v3 v4 v5 v6 v7
  poke _p (C'THCLongStorage v0 v1 v2 v3 v4 v5 v6 v7) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    pokeByteOff _p 48 v7
    return ()


{- typedef struct THCudaByteTensor {
            long * size;
            long * stride;
            int nDimension;
            THCByteStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaByteTensor; -}








data C'THCudaByteTensor = C'THCudaByteTensor{
  c'THCudaByteTensor'size :: Ptr CLong,
  c'THCudaByteTensor'stride :: Ptr CLong,
  c'THCudaByteTensor'nDimension :: CInt,
  c'THCudaByteTensor'storage :: Ptr C'THCByteStorage,
  c'THCudaByteTensor'storageOffset :: CLong,
  c'THCudaByteTensor'refcount :: CInt,
  c'THCudaByteTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaByteTensor'size p = plusPtr p 0
p'THCudaByteTensor'size :: Ptr (C'THCudaByteTensor) -> Ptr (Ptr CLong)
p'THCudaByteTensor'stride p = plusPtr p 8
p'THCudaByteTensor'stride :: Ptr (C'THCudaByteTensor) -> Ptr (Ptr CLong)
p'THCudaByteTensor'nDimension p = plusPtr p 16
p'THCudaByteTensor'nDimension :: Ptr (C'THCudaByteTensor) -> Ptr (CInt)
p'THCudaByteTensor'storage p = plusPtr p 24
p'THCudaByteTensor'storage :: Ptr (C'THCudaByteTensor) -> Ptr (Ptr C'THCByteStorage)
p'THCudaByteTensor'storageOffset p = plusPtr p 32
p'THCudaByteTensor'storageOffset :: Ptr (C'THCudaByteTensor) -> Ptr (CLong)
p'THCudaByteTensor'refcount p = plusPtr p 40
p'THCudaByteTensor'refcount :: Ptr (C'THCudaByteTensor) -> Ptr (CInt)
p'THCudaByteTensor'flag p = plusPtr p 44
p'THCudaByteTensor'flag :: Ptr (C'THCudaByteTensor) -> Ptr (CChar)
instance Storable C'THCudaByteTensor where
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
    return $ C'THCudaByteTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaByteTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaCharTensor {
            long * size;
            long * stride;
            int nDimension;
            THCCharStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaCharTensor; -}








data C'THCudaCharTensor = C'THCudaCharTensor{
  c'THCudaCharTensor'size :: Ptr CLong,
  c'THCudaCharTensor'stride :: Ptr CLong,
  c'THCudaCharTensor'nDimension :: CInt,
  c'THCudaCharTensor'storage :: Ptr C'THCCharStorage,
  c'THCudaCharTensor'storageOffset :: CLong,
  c'THCudaCharTensor'refcount :: CInt,
  c'THCudaCharTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaCharTensor'size p = plusPtr p 0
p'THCudaCharTensor'size :: Ptr (C'THCudaCharTensor) -> Ptr (Ptr CLong)
p'THCudaCharTensor'stride p = plusPtr p 8
p'THCudaCharTensor'stride :: Ptr (C'THCudaCharTensor) -> Ptr (Ptr CLong)
p'THCudaCharTensor'nDimension p = plusPtr p 16
p'THCudaCharTensor'nDimension :: Ptr (C'THCudaCharTensor) -> Ptr (CInt)
p'THCudaCharTensor'storage p = plusPtr p 24
p'THCudaCharTensor'storage :: Ptr (C'THCudaCharTensor) -> Ptr (Ptr C'THCCharStorage)
p'THCudaCharTensor'storageOffset p = plusPtr p 32
p'THCudaCharTensor'storageOffset :: Ptr (C'THCudaCharTensor) -> Ptr (CLong)
p'THCudaCharTensor'refcount p = plusPtr p 40
p'THCudaCharTensor'refcount :: Ptr (C'THCudaCharTensor) -> Ptr (CInt)
p'THCudaCharTensor'flag p = plusPtr p 44
p'THCudaCharTensor'flag :: Ptr (C'THCudaCharTensor) -> Ptr (CChar)
instance Storable C'THCudaCharTensor where
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
    return $ C'THCudaCharTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaCharTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaDoubleTensor {
            long * size;
            long * stride;
            int nDimension;
            THCDoubleStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaDoubleTensor; -}








data C'THCudaDoubleTensor = C'THCudaDoubleTensor{
  c'THCudaDoubleTensor'size :: Ptr CLong,
  c'THCudaDoubleTensor'stride :: Ptr CLong,
  c'THCudaDoubleTensor'nDimension :: CInt,
  c'THCudaDoubleTensor'storage :: Ptr C'THCDoubleStorage,
  c'THCudaDoubleTensor'storageOffset :: CLong,
  c'THCudaDoubleTensor'refcount :: CInt,
  c'THCudaDoubleTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaDoubleTensor'size p = plusPtr p 0
p'THCudaDoubleTensor'size :: Ptr (C'THCudaDoubleTensor) -> Ptr (Ptr CLong)
p'THCudaDoubleTensor'stride p = plusPtr p 8
p'THCudaDoubleTensor'stride :: Ptr (C'THCudaDoubleTensor) -> Ptr (Ptr CLong)
p'THCudaDoubleTensor'nDimension p = plusPtr p 16
p'THCudaDoubleTensor'nDimension :: Ptr (C'THCudaDoubleTensor) -> Ptr (CInt)
p'THCudaDoubleTensor'storage p = plusPtr p 24
p'THCudaDoubleTensor'storage :: Ptr (C'THCudaDoubleTensor) -> Ptr (Ptr C'THCDoubleStorage)
p'THCudaDoubleTensor'storageOffset p = plusPtr p 32
p'THCudaDoubleTensor'storageOffset :: Ptr (C'THCudaDoubleTensor) -> Ptr (CLong)
p'THCudaDoubleTensor'refcount p = plusPtr p 40
p'THCudaDoubleTensor'refcount :: Ptr (C'THCudaDoubleTensor) -> Ptr (CInt)
p'THCudaDoubleTensor'flag p = plusPtr p 44
p'THCudaDoubleTensor'flag :: Ptr (C'THCudaDoubleTensor) -> Ptr (CChar)
instance Storable C'THCudaDoubleTensor where
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
    return $ C'THCudaDoubleTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaDoubleTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaFloatTensor {
            long * size;
            long * stride;
            int nDimension;
            THCFloatStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaFloatTensor; -}








data C'THCudaFloatTensor = C'THCudaFloatTensor{
  c'THCudaFloatTensor'size :: Ptr CLong,
  c'THCudaFloatTensor'stride :: Ptr CLong,
  c'THCudaFloatTensor'nDimension :: CInt,
  c'THCudaFloatTensor'storage :: Ptr C'THCFloatStorage,
  c'THCudaFloatTensor'storageOffset :: CLong,
  c'THCudaFloatTensor'refcount :: CInt,
  c'THCudaFloatTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaFloatTensor'size p = plusPtr p 0
p'THCudaFloatTensor'size :: Ptr (C'THCudaFloatTensor) -> Ptr (Ptr CLong)
p'THCudaFloatTensor'stride p = plusPtr p 8
p'THCudaFloatTensor'stride :: Ptr (C'THCudaFloatTensor) -> Ptr (Ptr CLong)
p'THCudaFloatTensor'nDimension p = plusPtr p 16
p'THCudaFloatTensor'nDimension :: Ptr (C'THCudaFloatTensor) -> Ptr (CInt)
p'THCudaFloatTensor'storage p = plusPtr p 24
p'THCudaFloatTensor'storage :: Ptr (C'THCudaFloatTensor) -> Ptr (Ptr C'THCFloatStorage)
p'THCudaFloatTensor'storageOffset p = plusPtr p 32
p'THCudaFloatTensor'storageOffset :: Ptr (C'THCudaFloatTensor) -> Ptr (CLong)
p'THCudaFloatTensor'refcount p = plusPtr p 40
p'THCudaFloatTensor'refcount :: Ptr (C'THCudaFloatTensor) -> Ptr (CInt)
p'THCudaFloatTensor'flag p = plusPtr p 44
p'THCudaFloatTensor'flag :: Ptr (C'THCudaFloatTensor) -> Ptr (CChar)
instance Storable C'THCudaFloatTensor where
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
    return $ C'THCudaFloatTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaFloatTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaIntTensor {
            long * size;
            long * stride;
            int nDimension;
            THCIntStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaIntTensor; -}








data C'THCudaIntTensor = C'THCudaIntTensor{
  c'THCudaIntTensor'size :: Ptr CLong,
  c'THCudaIntTensor'stride :: Ptr CLong,
  c'THCudaIntTensor'nDimension :: CInt,
  c'THCudaIntTensor'storage :: Ptr C'THCIntStorage,
  c'THCudaIntTensor'storageOffset :: CLong,
  c'THCudaIntTensor'refcount :: CInt,
  c'THCudaIntTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaIntTensor'size p = plusPtr p 0
p'THCudaIntTensor'size :: Ptr (C'THCudaIntTensor) -> Ptr (Ptr CLong)
p'THCudaIntTensor'stride p = plusPtr p 8
p'THCudaIntTensor'stride :: Ptr (C'THCudaIntTensor) -> Ptr (Ptr CLong)
p'THCudaIntTensor'nDimension p = plusPtr p 16
p'THCudaIntTensor'nDimension :: Ptr (C'THCudaIntTensor) -> Ptr (CInt)
p'THCudaIntTensor'storage p = plusPtr p 24
p'THCudaIntTensor'storage :: Ptr (C'THCudaIntTensor) -> Ptr (Ptr C'THCIntStorage)
p'THCudaIntTensor'storageOffset p = plusPtr p 32
p'THCudaIntTensor'storageOffset :: Ptr (C'THCudaIntTensor) -> Ptr (CLong)
p'THCudaIntTensor'refcount p = plusPtr p 40
p'THCudaIntTensor'refcount :: Ptr (C'THCudaIntTensor) -> Ptr (CInt)
p'THCudaIntTensor'flag p = plusPtr p 44
p'THCudaIntTensor'flag :: Ptr (C'THCudaIntTensor) -> Ptr (CChar)
instance Storable C'THCudaIntTensor where
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
    return $ C'THCudaIntTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaIntTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaShortTensor {
            long * size;
            long * stride;
            int nDimension;
            THCShortStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaShortTensor; -}








data C'THCudaShortTensor = C'THCudaShortTensor{
  c'THCudaShortTensor'size :: Ptr CLong,
  c'THCudaShortTensor'stride :: Ptr CLong,
  c'THCudaShortTensor'nDimension :: CInt,
  c'THCudaShortTensor'storage :: Ptr C'THCShortStorage,
  c'THCudaShortTensor'storageOffset :: CLong,
  c'THCudaShortTensor'refcount :: CInt,
  c'THCudaShortTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaShortTensor'size p = plusPtr p 0
p'THCudaShortTensor'size :: Ptr (C'THCudaShortTensor) -> Ptr (Ptr CLong)
p'THCudaShortTensor'stride p = plusPtr p 8
p'THCudaShortTensor'stride :: Ptr (C'THCudaShortTensor) -> Ptr (Ptr CLong)
p'THCudaShortTensor'nDimension p = plusPtr p 16
p'THCudaShortTensor'nDimension :: Ptr (C'THCudaShortTensor) -> Ptr (CInt)
p'THCudaShortTensor'storage p = plusPtr p 24
p'THCudaShortTensor'storage :: Ptr (C'THCudaShortTensor) -> Ptr (Ptr C'THCShortStorage)
p'THCudaShortTensor'storageOffset p = plusPtr p 32
p'THCudaShortTensor'storageOffset :: Ptr (C'THCudaShortTensor) -> Ptr (CLong)
p'THCudaShortTensor'refcount p = plusPtr p 40
p'THCudaShortTensor'refcount :: Ptr (C'THCudaShortTensor) -> Ptr (CInt)
p'THCudaShortTensor'flag p = plusPtr p 44
p'THCudaShortTensor'flag :: Ptr (C'THCudaShortTensor) -> Ptr (CChar)
instance Storable C'THCudaShortTensor where
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
    return $ C'THCudaShortTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaShortTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


{- typedef struct THCudaLongTensor {
            long * size;
            long * stride;
            int nDimension;
            THCLongStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCudaLongTensor; -}








data C'THCudaLongTensor = C'THCudaLongTensor{
  c'THCudaLongTensor'size :: Ptr CLong,
  c'THCudaLongTensor'stride :: Ptr CLong,
  c'THCudaLongTensor'nDimension :: CInt,
  c'THCudaLongTensor'storage :: Ptr C'THCLongStorage,
  c'THCudaLongTensor'storageOffset :: CLong,
  c'THCudaLongTensor'refcount :: CInt,
  c'THCudaLongTensor'flag :: CChar
} deriving (Eq,Show)
p'THCudaLongTensor'size p = plusPtr p 0
p'THCudaLongTensor'size :: Ptr (C'THCudaLongTensor) -> Ptr (Ptr CLong)
p'THCudaLongTensor'stride p = plusPtr p 8
p'THCudaLongTensor'stride :: Ptr (C'THCudaLongTensor) -> Ptr (Ptr CLong)
p'THCudaLongTensor'nDimension p = plusPtr p 16
p'THCudaLongTensor'nDimension :: Ptr (C'THCudaLongTensor) -> Ptr (CInt)
p'THCudaLongTensor'storage p = plusPtr p 24
p'THCudaLongTensor'storage :: Ptr (C'THCudaLongTensor) -> Ptr (Ptr C'THCLongStorage)
p'THCudaLongTensor'storageOffset p = plusPtr p 32
p'THCudaLongTensor'storageOffset :: Ptr (C'THCudaLongTensor) -> Ptr (CLong)
p'THCudaLongTensor'refcount p = plusPtr p 40
p'THCudaLongTensor'refcount :: Ptr (C'THCudaLongTensor) -> Ptr (CInt)
p'THCudaLongTensor'flag p = plusPtr p 44
p'THCudaLongTensor'flag :: Ptr (C'THCudaLongTensor) -> Ptr (CChar)
instance Storable C'THCudaLongTensor where
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
    return $ C'THCudaLongTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCudaLongTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()


