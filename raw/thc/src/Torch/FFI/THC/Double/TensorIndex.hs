{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorIndex
  ( c_indexCopy
  , c_indexAdd
  , c_indexFill
  , c_indexSelect
  , c_take
  , c_put
  , c_indexCopy_long
  , c_indexAdd_long
  , c_indexFill_long
  , c_indexSelect_long
  , c_calculateAdvancedIndexingOffsets
  , p_indexCopy
  , p_indexAdd
  , p_indexFill
  , p_indexSelect
  , p_take
  , p_put
  , p_indexCopy_long
  , p_indexAdd_long
  , p_indexFill_long
  , p_indexSelect_long
  , p_calculateAdvancedIndexingOffsets
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_indexCopy :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexCopy"
  c_indexCopy :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexAdd"
  c_indexAdd :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexFill"
  c_indexFill :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ()

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexSelect"
  c_indexSelect :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_take"
  c_take :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_put"
  c_put :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CInt -> IO ()

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexCopy_long"
  c_indexCopy_long :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexAdd_long"
  c_indexAdd_long :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexFill_long"
  c_indexFill_long :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ()

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_indexSelect_long"
  c_indexSelect_long :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THCDoubleTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CPtrdiff -> Ptr (Ptr CTHCudaLongTensor) -> IO ()

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexFill"
  p_indexFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ())

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_take"
  p_take :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_put"
  p_put :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CInt -> IO ())

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ())

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THCDoubleTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CPtrdiff -> Ptr (Ptr CTHCudaLongTensor) -> IO ())