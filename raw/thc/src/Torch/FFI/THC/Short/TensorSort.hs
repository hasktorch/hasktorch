{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorSort where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_sortKeyValueInplace :  state keys values dim order -> void
foreign import ccall "THCTensorSort.h THCShortTensor_sortKeyValueInplace"
  c_sortKeyValueInplace :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> CInt -> CInt -> IO ()

-- | c_sort :  state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h THCShortTensor_sort"
  c_sort :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> CInt -> CInt -> IO ()

-- | p_sortKeyValueInplace : Pointer to function : state keys values dim order -> void
foreign import ccall "THCTensorSort.h &THCShortTensor_sortKeyValueInplace"
  p_sortKeyValueInplace :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> CInt -> CInt -> IO ())

-- | p_sort : Pointer to function : state sorted indices input dim order -> void
foreign import ccall "THCTensorSort.h &THCShortTensor_sort"
  p_sort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> CInt -> CInt -> IO ())