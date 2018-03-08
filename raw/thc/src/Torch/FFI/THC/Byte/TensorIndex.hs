{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorIndex
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
foreign import ccall "THCTensorIndex.h THByteTensor_indexCopy"
  c_indexCopy :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexAdd"
  c_indexAdd :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexFill"
  c_indexFill :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (())

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexSelect"
  c_indexSelect :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THByteTensor_take"
  c_take :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THByteTensor_put"
  c_put :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexCopy_long"
  c_indexCopy_long :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexAdd_long"
  c_indexAdd_long :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexFill_long"
  c_indexFill_long :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (())

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THByteTensor_indexSelect_long"
  c_indexSelect_long :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THByteTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (())

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (()))

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_take"
  p_take :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_put"
  p_put :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (()))

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THByteTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (()))