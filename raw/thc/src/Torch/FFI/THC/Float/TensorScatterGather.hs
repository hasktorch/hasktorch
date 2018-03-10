{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorScatterGather
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
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_gather"
  c_gather :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatter"
  c_scatter :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatterAdd"
  c_scatterAdd :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatterFill"
  c_scatterFill :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> CFloat -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_gather"
  p_gather :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatter"
  p_scatter :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> Ptr CTHCudaLongTensor -> CFloat -> IO ())