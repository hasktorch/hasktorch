{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.General where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_THCState_alloc :   -> THCState *
foreign import ccall "THCGeneral.h THCState_alloc"
  c_THCState_alloc :: IO (Ptr C'THCState)

-- | c_THCState_free :  state -> void
foreign import ccall "THCGeneral.h THCState_free"
  c_THCState_free :: Ptr C'THCState -> IO ()

-- | c_THCudaInit :  state -> void
foreign import ccall "THCGeneral.h THCudaInit"
  c_THCudaInit :: Ptr C'THCState -> IO ()

-- | c_THCudaShutdown :  state -> void
foreign import ccall "THCGeneral.h THCudaShutdown"
  c_THCudaShutdown :: Ptr C'THCState -> IO ()

-- | c_THCState_getPeerToPeerAccess :  state dev devToAccess -> int
foreign import ccall "THCGeneral.h THCState_getPeerToPeerAccess"
  c_THCState_getPeerToPeerAccess :: Ptr C'THCState -> CInt -> CInt -> IO CInt

-- | c_THCState_setPeerToPeerAccess :  state dev devToAccess enable -> void
foreign import ccall "THCGeneral.h THCState_setPeerToPeerAccess"
  c_THCState_setPeerToPeerAccess :: Ptr C'THCState -> CInt -> CInt -> CInt -> IO ()

-- | c_THCState_getKernelPeerToPeerAccessEnabled :  state -> int
foreign import ccall "THCGeneral.h THCState_getKernelPeerToPeerAccessEnabled"
  c_THCState_getKernelPeerToPeerAccessEnabled :: Ptr C'THCState -> IO CInt

-- | c_THCState_setKernelPeerToPeerAccessEnabled :  state val -> void
foreign import ccall "THCGeneral.h THCState_setKernelPeerToPeerAccessEnabled"
  c_THCState_setKernelPeerToPeerAccessEnabled :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_getCudaHostAllocator :  state -> THAllocator *
foreign import ccall "THCGeneral.h THCState_getCudaHostAllocator"
  c_THCState_getCudaHostAllocator :: Ptr C'THCState -> IO (Ptr C'THAllocator)

-- | c_THCState_getCudaUVAAllocator :  state -> THAllocator *
foreign import ccall "THCGeneral.h THCState_getCudaUVAAllocator"
  c_THCState_getCudaUVAAllocator :: Ptr C'THCState -> IO (Ptr C'THAllocator)

-- | c_THCState_isCachingAllocatorEnabled :  state -> int
foreign import ccall "THCGeneral.h THCState_isCachingAllocatorEnabled"
  c_THCState_isCachingAllocatorEnabled :: Ptr C'THCState -> IO CInt

-- | c_THCMagma_init :  state -> void
foreign import ccall "THCGeneral.h THCMagma_init"
  c_THCMagma_init :: Ptr C'THCState -> IO ()

-- | c_THCState_getNumDevices :  state -> int
foreign import ccall "THCGeneral.h THCState_getNumDevices"
  c_THCState_getNumDevices :: Ptr C'THCState -> IO CInt

-- | c_THCState_reserveStreams :  state numStreams nonBlocking -> void
foreign import ccall "THCGeneral.h THCState_reserveStreams"
  c_THCState_reserveStreams :: Ptr C'THCState -> CInt -> CInt -> IO ()

-- | c_THCState_getNumStreams :  state -> int
foreign import ccall "THCGeneral.h THCState_getNumStreams"
  c_THCState_getNumStreams :: Ptr C'THCState -> IO CInt

-- | c_THCState_getStream :  state -> THCStream *
foreign import ccall "THCGeneral.h THCState_getStream"
  c_THCState_getStream :: Ptr C'THCState -> IO (Ptr C'THCStream)

-- | c_THCState_setStream :  state stream -> void
foreign import ccall "THCGeneral.h THCState_setStream"
  c_THCState_setStream :: Ptr C'THCState -> Ptr C'THCStream -> IO ()

-- | c_THCState_getCurrentStreamIndex :  state -> int
foreign import ccall "THCGeneral.h THCState_getCurrentStreamIndex"
  c_THCState_getCurrentStreamIndex :: Ptr C'THCState -> IO CInt

-- | c_THCState_setCurrentStreamIndex :  state stream -> void
foreign import ccall "THCGeneral.h THCState_setCurrentStreamIndex"
  c_THCState_setCurrentStreamIndex :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_reserveBlasHandles :  state numHandles -> void
foreign import ccall "THCGeneral.h THCState_reserveBlasHandles"
  c_THCState_reserveBlasHandles :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_getNumBlasHandles :  state -> int
foreign import ccall "THCGeneral.h THCState_getNumBlasHandles"
  c_THCState_getNumBlasHandles :: Ptr C'THCState -> IO CInt

-- | c_THCState_reserveSparseHandles :  state numHandles -> void
foreign import ccall "THCGeneral.h THCState_reserveSparseHandles"
  c_THCState_reserveSparseHandles :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_getNumSparseHandles :  state -> int
foreign import ccall "THCGeneral.h THCState_getNumSparseHandles"
  c_THCState_getNumSparseHandles :: Ptr C'THCState -> IO CInt

-- | c_THCState_getCurrentBlasHandleIndex :  state -> int
foreign import ccall "THCGeneral.h THCState_getCurrentBlasHandleIndex"
  c_THCState_getCurrentBlasHandleIndex :: Ptr C'THCState -> IO CInt

-- | c_THCState_setCurrentBlasHandleIndex :  state handle -> void
foreign import ccall "THCGeneral.h THCState_setCurrentBlasHandleIndex"
  c_THCState_setCurrentBlasHandleIndex :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_getCurrentSparseHandleIndex :  state -> int
foreign import ccall "THCGeneral.h THCState_getCurrentSparseHandleIndex"
  c_THCState_getCurrentSparseHandleIndex :: Ptr C'THCState -> IO CInt

-- | c_THCState_setCurrentSparseHandleIndex :  state handle -> void
foreign import ccall "THCGeneral.h THCState_setCurrentSparseHandleIndex"
  c_THCState_setCurrentSparseHandleIndex :: Ptr C'THCState -> CInt -> IO ()

-- | c_THCState_getCurrentDeviceScratchSpace :  state -> void *
foreign import ccall "THCGeneral.h THCState_getCurrentDeviceScratchSpace"
  c_THCState_getCurrentDeviceScratchSpace :: Ptr C'THCState -> IO (Ptr ())

-- | c_THCState_getDeviceScratchSpace :  state device stream -> void *
foreign import ccall "THCGeneral.h THCState_getDeviceScratchSpace"
  c_THCState_getDeviceScratchSpace :: Ptr C'THCState -> CInt -> CInt -> IO (Ptr ())

-- | c_THCState_getCurrentDeviceScratchSpaceSize :  state -> size_t
foreign import ccall "THCGeneral.h THCState_getCurrentDeviceScratchSpaceSize"
  c_THCState_getCurrentDeviceScratchSpaceSize :: Ptr C'THCState -> IO CSize

-- | c_THCState_getDeviceScratchSpaceSize :  state device -> size_t
foreign import ccall "THCGeneral.h THCState_getDeviceScratchSpaceSize"
  c_THCState_getDeviceScratchSpaceSize :: Ptr C'THCState -> CInt -> IO CSize

-- | c_THCudaHostAlloc :  state size -> void *
foreign import ccall "THCGeneral.h THCudaHostAlloc"
  c_THCudaHostAlloc :: Ptr C'THCState -> CSize -> IO (Ptr ())

-- | c_THCudaHostFree :  state ptr -> void
foreign import ccall "THCGeneral.h THCudaHostFree"
  c_THCudaHostFree :: Ptr C'THCState -> Ptr () -> IO ()

-- | c_THCudaHostRecord :  state ptr -> void
foreign import ccall "THCGeneral.h THCudaHostRecord"
  c_THCudaHostRecord :: Ptr C'THCState -> Ptr () -> IO ()

-- | p_THCState_alloc : Pointer to function :  -> THCState *
foreign import ccall "THCGeneral.h &THCState_alloc"
  p_THCState_alloc :: FunPtr (IO (Ptr C'THCState))

-- | p_THCState_free : Pointer to function : state -> void
foreign import ccall "THCGeneral.h &THCState_free"
  p_THCState_free :: FunPtr (Ptr C'THCState -> IO ())

-- | p_THCudaInit : Pointer to function : state -> void
foreign import ccall "THCGeneral.h &THCudaInit"
  p_THCudaInit :: FunPtr (Ptr C'THCState -> IO ())

-- | p_THCudaShutdown : Pointer to function : state -> void
foreign import ccall "THCGeneral.h &THCudaShutdown"
  p_THCudaShutdown :: FunPtr (Ptr C'THCState -> IO ())

-- | p_THCState_getPeerToPeerAccess : Pointer to function : state dev devToAccess -> int
foreign import ccall "THCGeneral.h &THCState_getPeerToPeerAccess"
  p_THCState_getPeerToPeerAccess :: FunPtr (Ptr C'THCState -> CInt -> CInt -> IO CInt)

-- | p_THCState_setPeerToPeerAccess : Pointer to function : state dev devToAccess enable -> void
foreign import ccall "THCGeneral.h &THCState_setPeerToPeerAccess"
  p_THCState_setPeerToPeerAccess :: FunPtr (Ptr C'THCState -> CInt -> CInt -> CInt -> IO ())

-- | p_THCState_getKernelPeerToPeerAccessEnabled : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getKernelPeerToPeerAccessEnabled"
  p_THCState_getKernelPeerToPeerAccessEnabled :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_setKernelPeerToPeerAccessEnabled : Pointer to function : state val -> void
foreign import ccall "THCGeneral.h &THCState_setKernelPeerToPeerAccessEnabled"
  p_THCState_setKernelPeerToPeerAccessEnabled :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_getCudaHostAllocator : Pointer to function : state -> THAllocator *
foreign import ccall "THCGeneral.h &THCState_getCudaHostAllocator"
  p_THCState_getCudaHostAllocator :: FunPtr (Ptr C'THCState -> IO (Ptr C'THAllocator))

-- | p_THCState_getCudaUVAAllocator : Pointer to function : state -> THAllocator *
foreign import ccall "THCGeneral.h &THCState_getCudaUVAAllocator"
  p_THCState_getCudaUVAAllocator :: FunPtr (Ptr C'THCState -> IO (Ptr C'THAllocator))

-- | p_THCState_isCachingAllocatorEnabled : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_isCachingAllocatorEnabled"
  p_THCState_isCachingAllocatorEnabled :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCMagma_init : Pointer to function : state -> void
foreign import ccall "THCGeneral.h &THCMagma_init"
  p_THCMagma_init :: FunPtr (Ptr C'THCState -> IO ())

-- | p_THCState_getNumDevices : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getNumDevices"
  p_THCState_getNumDevices :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_reserveStreams : Pointer to function : state numStreams nonBlocking -> void
foreign import ccall "THCGeneral.h &THCState_reserveStreams"
  p_THCState_reserveStreams :: FunPtr (Ptr C'THCState -> CInt -> CInt -> IO ())

-- | p_THCState_getNumStreams : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getNumStreams"
  p_THCState_getNumStreams :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_getStream : Pointer to function : state -> THCStream *
foreign import ccall "THCGeneral.h &THCState_getStream"
  p_THCState_getStream :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCStream))

-- | p_THCState_setStream : Pointer to function : state stream -> void
foreign import ccall "THCGeneral.h &THCState_setStream"
  p_THCState_setStream :: FunPtr (Ptr C'THCState -> Ptr C'THCStream -> IO ())

-- | p_THCState_getCurrentStreamIndex : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getCurrentStreamIndex"
  p_THCState_getCurrentStreamIndex :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_setCurrentStreamIndex : Pointer to function : state stream -> void
foreign import ccall "THCGeneral.h &THCState_setCurrentStreamIndex"
  p_THCState_setCurrentStreamIndex :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_reserveBlasHandles : Pointer to function : state numHandles -> void
foreign import ccall "THCGeneral.h &THCState_reserveBlasHandles"
  p_THCState_reserveBlasHandles :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_getNumBlasHandles : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getNumBlasHandles"
  p_THCState_getNumBlasHandles :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_reserveSparseHandles : Pointer to function : state numHandles -> void
foreign import ccall "THCGeneral.h &THCState_reserveSparseHandles"
  p_THCState_reserveSparseHandles :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_getNumSparseHandles : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getNumSparseHandles"
  p_THCState_getNumSparseHandles :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_getCurrentBlasHandleIndex : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getCurrentBlasHandleIndex"
  p_THCState_getCurrentBlasHandleIndex :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_setCurrentBlasHandleIndex : Pointer to function : state handle -> void
foreign import ccall "THCGeneral.h &THCState_setCurrentBlasHandleIndex"
  p_THCState_setCurrentBlasHandleIndex :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_getCurrentSparseHandleIndex : Pointer to function : state -> int
foreign import ccall "THCGeneral.h &THCState_getCurrentSparseHandleIndex"
  p_THCState_getCurrentSparseHandleIndex :: FunPtr (Ptr C'THCState -> IO CInt)

-- | p_THCState_setCurrentSparseHandleIndex : Pointer to function : state handle -> void
foreign import ccall "THCGeneral.h &THCState_setCurrentSparseHandleIndex"
  p_THCState_setCurrentSparseHandleIndex :: FunPtr (Ptr C'THCState -> CInt -> IO ())

-- | p_THCState_getCurrentDeviceScratchSpace : Pointer to function : state -> void *
foreign import ccall "THCGeneral.h &THCState_getCurrentDeviceScratchSpace"
  p_THCState_getCurrentDeviceScratchSpace :: FunPtr (Ptr C'THCState -> IO (Ptr ()))

-- | p_THCState_getDeviceScratchSpace : Pointer to function : state device stream -> void *
foreign import ccall "THCGeneral.h &THCState_getDeviceScratchSpace"
  p_THCState_getDeviceScratchSpace :: FunPtr (Ptr C'THCState -> CInt -> CInt -> IO (Ptr ()))

-- | p_THCState_getCurrentDeviceScratchSpaceSize : Pointer to function : state -> size_t
foreign import ccall "THCGeneral.h &THCState_getCurrentDeviceScratchSpaceSize"
  p_THCState_getCurrentDeviceScratchSpaceSize :: FunPtr (Ptr C'THCState -> IO CSize)

-- | p_THCState_getDeviceScratchSpaceSize : Pointer to function : state device -> size_t
foreign import ccall "THCGeneral.h &THCState_getDeviceScratchSpaceSize"
  p_THCState_getDeviceScratchSpaceSize :: FunPtr (Ptr C'THCState -> CInt -> IO CSize)

-- | p_THCudaHostAlloc : Pointer to function : state size -> void *
foreign import ccall "THCGeneral.h &THCudaHostAlloc"
  p_THCudaHostAlloc :: FunPtr (Ptr C'THCState -> CSize -> IO (Ptr ()))

-- | p_THCudaHostFree : Pointer to function : state ptr -> void
foreign import ccall "THCGeneral.h &THCudaHostFree"
  p_THCudaHostFree :: FunPtr (Ptr C'THCState -> Ptr () -> IO ())

-- | p_THCudaHostRecord : Pointer to function : state ptr -> void
foreign import ccall "THCGeneral.h &THCudaHostRecord"
  p_THCudaHostRecord :: FunPtr (Ptr C'THCState -> Ptr () -> IO ())