{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THLongStorage_sizeDesc :  size -> THDescBuff
foreign import ccall "THStorage.h THLongStorage_sizeDesc"
  c_THLongStorage_sizeDesc :: Ptr C'THLongStorage -> IO (Ptr C'THDescBuff)

-- | c_THLongStorage_newInferSize :  size nElement -> THLongStorage *
foreign import ccall "THStorage.h THLongStorage_newInferSize"
  c_THLongStorage_newInferSize :: Ptr C'THLongStorage -> CPtrdiff -> IO (Ptr C'THLongStorage)

-- | c_THLongStorage_inferSize2 :  output sizesA dimsA sizesB dimsB error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSize2"
  c_THLongStorage_inferSize2 :: Ptr C'THLongStorage -> Ptr CLLong -> CLLong -> Ptr CLLong -> CLLong -> Ptr CChar -> CInt -> IO CInt

-- | c_THLongStorage_inferSizeN :  output n sizes dims error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSizeN"
  c_THLongStorage_inferSizeN :: Ptr C'THLongStorage -> CInt -> Ptr (Ptr CLLong) -> Ptr CLLong -> Ptr CChar -> CInt -> IO CInt

-- | c_THLongStorage_inferExpandGeometry :  tensorSizes tensorStrides tensorDim sizes expandedSizes expandedStrides error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferExpandGeometry"
  c_THLongStorage_inferExpandGeometry :: Ptr CLLong -> Ptr CLLong -> CLLong -> Ptr C'THLongStorage -> Ptr (Ptr CLLong) -> Ptr (Ptr CLLong) -> Ptr CChar -> CInt -> IO CInt

-- | p_THLongStorage_sizeDesc : Pointer to function : size -> THDescBuff
foreign import ccall "THStorage.h &THLongStorage_sizeDesc"
  p_THLongStorage_sizeDesc :: FunPtr (Ptr C'THLongStorage -> IO (Ptr C'THDescBuff))

-- | p_THLongStorage_newInferSize : Pointer to function : size nElement -> THLongStorage *
foreign import ccall "THStorage.h &THLongStorage_newInferSize"
  p_THLongStorage_newInferSize :: FunPtr (Ptr C'THLongStorage -> CPtrdiff -> IO (Ptr C'THLongStorage))

-- | p_THLongStorage_inferSize2 : Pointer to function : output sizesA dimsA sizesB dimsB error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferSize2"
  p_THLongStorage_inferSize2 :: FunPtr (Ptr C'THLongStorage -> Ptr CLLong -> CLLong -> Ptr CLLong -> CLLong -> Ptr CChar -> CInt -> IO CInt)

-- | p_THLongStorage_inferSizeN : Pointer to function : output n sizes dims error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferSizeN"
  p_THLongStorage_inferSizeN :: FunPtr (Ptr C'THLongStorage -> CInt -> Ptr (Ptr CLLong) -> Ptr CLLong -> Ptr CChar -> CInt -> IO CInt)

-- | p_THLongStorage_inferExpandGeometry : Pointer to function : tensorSizes tensorStrides tensorDim sizes expandedSizes expandedStrides error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferExpandGeometry"
  p_THLongStorage_inferExpandGeometry :: FunPtr (Ptr CLLong -> Ptr CLLong -> CLLong -> Ptr C'THLongStorage -> Ptr (Ptr CLLong) -> Ptr (Ptr CLLong) -> Ptr CChar -> CInt -> IO CInt)