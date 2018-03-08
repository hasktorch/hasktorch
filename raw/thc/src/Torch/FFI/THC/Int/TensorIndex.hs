{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorIndex
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
foreign import ccall "THCTensorIndex.h THIntTensor_indexCopy"
  c_indexCopy :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexAdd"
  c_indexAdd :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexFill"
  c_indexFill :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexSelect"
  c_indexSelect :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THIntTensor_take"
  c_take :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THIntTensor_put"
  c_put :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexCopy_long"
  c_indexCopy_long :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexAdd_long"
  c_indexAdd_long :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexFill_long"
  c_indexFill_long :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THIntTensor_indexSelect_long"
  c_indexSelect_long :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THIntTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (())

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_take"
  p_take :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_put"
  p_put :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THIntTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (()))