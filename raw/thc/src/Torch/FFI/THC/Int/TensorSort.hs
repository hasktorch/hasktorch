{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorSort
  ( c_sortKeyValueInplace
  , c_sort
  , p_sortKeyValueInplace
  , p_sort
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_sortKeyValueInplace :  state keys values dim order -> void
foreign import ccall "THCTensorSort.h THIntTensor_sortKeyValueInplace"
  c_sortKeyValueInplace :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_sort :  state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h THIntTensor_sort"
  c_sort :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | p_sortKeyValueInplace : Pointer to function : state keys values dim order -> void
foreign import ccall "THCTensorSort.h &THIntTensor_sortKeyValueInplace"
  p_sortKeyValueInplace :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_sort : Pointer to function : state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h &THIntTensor_sort"
  p_sort :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))