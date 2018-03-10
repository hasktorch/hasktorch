{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorIndex where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_indexCopy :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexCopy"
  c_indexCopy :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexAdd"
  c_indexAdd :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexFill"
  c_indexFill :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> CTHHalf -> IO ()

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexSelect"
  c_indexSelect :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_take"
  c_take :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_put"
  c_put :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ()

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexCopy_long"
  c_indexCopy_long :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexAdd_long"
  c_indexAdd_long :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexFill_long"
  c_indexFill_long :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> CTHHalf -> IO ()

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_indexSelect_long"
  c_indexSelect_long :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THCHalfTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CPtrdiff -> Ptr (Ptr CTHCudaLongTensor) -> IO ()

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexFill"
  p_indexFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> CTHHalf -> IO ())

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_take"
  p_take :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_put"
  p_put :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ())

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> CTHHalf -> IO ())

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THCHalfTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CPtrdiff -> Ptr (Ptr CTHCudaLongTensor) -> IO ())