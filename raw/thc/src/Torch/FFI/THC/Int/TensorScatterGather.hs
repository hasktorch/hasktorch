{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorScatterGather
  ( c_gather
  , c_scatter
  , c_scatterAdd
  , c_scatterFill
  , p_gather
  , p_scatter
  , p_scatterAdd
  , p_scatterFill
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gather :  state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h THCIntTensor_gather"
  c_gather :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCIntTensor_scatter"
  c_scatter :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCIntTensor_scatterAdd"
  c_scatterAdd :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCIntTensor_scatterFill"
  c_scatterFill :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> CInt -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCIntTensor_gather"
  p_gather :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCIntTensor_scatter"
  p_scatter :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCIntTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCIntTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaLongTensor -> CInt -> IO ())