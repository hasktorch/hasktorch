{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorIndex
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
foreign import ccall "THCTensorIndex.h THShortTensor_indexCopy"
  c_indexCopy :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexAdd"
  c_indexAdd :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexFill"
  c_indexFill :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> CShort -> IO (())

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexSelect"
  c_indexSelect :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THShortTensor_take"
  c_take :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THShortTensor_put"
  c_put :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (())

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexCopy_long"
  c_indexCopy_long :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexAdd_long"
  c_indexAdd_long :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexFill_long"
  c_indexFill_long :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> CShort -> IO (())

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THShortTensor_indexSelect_long"
  c_indexSelect_long :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THShortTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (())

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> CShort -> IO (()))

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_take"
  p_take :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_put"
  p_put :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (()))

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> CShort -> IO (()))

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THShortTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CPtrdiff -> Ptr (Ptr (CTHLongTensor)) -> IO (()))