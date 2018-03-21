{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorIndex where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_indexCopy :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexCopy"
  c_indexCopy :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_indexAdd :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexAdd"
  c_indexAdd :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_indexFill :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexFill"
  c_indexFill :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> CDouble -> IO ()

-- | c_indexSelect :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexSelect"
  c_indexSelect :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ()

-- | c_take :  state res_ src index -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_take"
  c_take :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> IO ()

-- | c_put :  state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_put"
  c_put :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_indexCopy_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexCopy_long"
  c_indexCopy_long :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_indexAdd_long :  state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexAdd_long"
  c_indexAdd_long :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_indexFill_long :  state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexFill_long"
  c_indexFill_long :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ()

-- | c_indexSelect_long :  state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_indexSelect_long"
  c_indexSelect_long :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | c_calculateAdvancedIndexingOffsets :  state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h THCudaDoubleTensor_calculateAdvancedIndexingOffsets"
  c_calculateAdvancedIndexingOffsets :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CPtrdiff -> Ptr (Ptr C'THCudaLongTensor) -> IO ()

-- | p_indexCopy : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_indexAdd : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_indexFill : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexFill"
  p_indexFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> CDouble -> IO ())

-- | p_indexSelect : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ())

-- | p_take : Pointer to function : state res_ src index -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_take"
  p_take :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> IO ())

-- | p_put : Pointer to function : state res_ indices src accumulate -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_put"
  p_put :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_indexCopy_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexCopy_long"
  p_indexCopy_long :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_indexAdd_long : Pointer to function : state res_ dim indices src -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexAdd_long"
  p_indexAdd_long :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_indexFill_long : Pointer to function : state tensor dim index val -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexFill_long"
  p_indexFill_long :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ())

-- | p_indexSelect_long : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_indexSelect_long"
  p_indexSelect_long :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_calculateAdvancedIndexingOffsets : Pointer to function : state output indexed baseOffset indexers -> void
foreign import ccall "THCTensorIndex.h &THCudaDoubleTensor_calculateAdvancedIndexingOffsets"
  p_calculateAdvancedIndexingOffsets :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CPtrdiff -> Ptr (Ptr C'THCudaLongTensor) -> IO ())