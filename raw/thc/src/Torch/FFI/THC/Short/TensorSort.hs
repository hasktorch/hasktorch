{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorSort
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
foreign import ccall "THCTensorSort.h THShortTensor_sortKeyValueInplace"
  c_sortKeyValueInplace :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_sort :  state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h THShortTensor_sort"
  c_sort :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | p_sortKeyValueInplace : Pointer to function : state keys values dim order -> void
foreign import ccall "THCTensorSort.h &THShortTensor_sortKeyValueInplace"
  p_sortKeyValueInplace :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_sort : Pointer to function : state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h &THShortTensor_sort"
  p_sort :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))