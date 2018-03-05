{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Storage
  ( c_THLongStorage_sizeDesc
  , c_THLongStorage_newInferSize
  , c_THLongStorage_inferSize2
  , c_THLongStorage_inferSizeN
  , c_THLongStorage_inferExpandGeometry
  , p_THLongStorage_sizeDesc
  , p_THLongStorage_newInferSize
  , p_THLongStorage_inferSize2
  , p_THLongStorage_inferSizeN
  , p_THLongStorage_inferExpandGeometry
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THLongStorage_sizeDesc :  size -> THDescBuff
foreign import ccall "THStorage.h THLongStorage_sizeDesc"
  c_THLongStorage_sizeDesc :: Ptr (CTHLongStorage) -> IO (CTHDescBuff)

-- | c_THLongStorage_newInferSize :  size nElement -> THLongStorage *
foreign import ccall "THStorage.h THLongStorage_newInferSize"
  c_THLongStorage_newInferSize :: Ptr (CTHLongStorage) -> CPtrdiff -> IO (Ptr (CTHLongStorage))

-- | c_THLongStorage_inferSize2 :  output sizesA dimsA sizesB dimsB error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSize2"
  c_THLongStorage_inferSize2 :: Ptr (CTHLongStorage) -> Ptr (CLLong) -> CLLong -> Ptr (CLLong) -> CLLong -> Ptr (CChar) -> CInt -> IO (CInt)

-- | c_THLongStorage_inferSizeN :  output n sizes dims error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSizeN"
  c_THLongStorage_inferSizeN :: Ptr (CTHLongStorage) -> CInt -> Ptr (Ptr (CLLong)) -> Ptr (CLLong) -> Ptr (CChar) -> CInt -> IO (CInt)

-- | c_THLongStorage_inferExpandGeometry :  tensorSizes tensorStrides tensorDim sizes expandedSizes expandedStrides error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferExpandGeometry"
  c_THLongStorage_inferExpandGeometry :: Ptr (CLLong) -> Ptr (CLLong) -> CLLong -> Ptr (CTHLongStorage) -> Ptr (Ptr (CLLong)) -> Ptr (Ptr (CLLong)) -> Ptr (CChar) -> CInt -> IO (CInt)

-- | p_THLongStorage_sizeDesc : Pointer to function : size -> THDescBuff
foreign import ccall "THStorage.h &THLongStorage_sizeDesc"
  p_THLongStorage_sizeDesc :: FunPtr (Ptr (CTHLongStorage) -> IO (CTHDescBuff))

-- | p_THLongStorage_newInferSize : Pointer to function : size nElement -> THLongStorage *
foreign import ccall "THStorage.h &THLongStorage_newInferSize"
  p_THLongStorage_newInferSize :: FunPtr (Ptr (CTHLongStorage) -> CPtrdiff -> IO (Ptr (CTHLongStorage)))

-- | p_THLongStorage_inferSize2 : Pointer to function : output sizesA dimsA sizesB dimsB error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferSize2"
  p_THLongStorage_inferSize2 :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CLLong) -> CLLong -> Ptr (CLLong) -> CLLong -> Ptr (CChar) -> CInt -> IO (CInt))

-- | p_THLongStorage_inferSizeN : Pointer to function : output n sizes dims error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferSizeN"
  p_THLongStorage_inferSizeN :: FunPtr (Ptr (CTHLongStorage) -> CInt -> Ptr (Ptr (CLLong)) -> Ptr (CLLong) -> Ptr (CChar) -> CInt -> IO (CInt))

-- | p_THLongStorage_inferExpandGeometry : Pointer to function : tensorSizes tensorStrides tensorDim sizes expandedSizes expandedStrides error_buffer buffer_len -> int
foreign import ccall "THStorage.h &THLongStorage_inferExpandGeometry"
  p_THLongStorage_inferExpandGeometry :: FunPtr (Ptr (CLLong) -> Ptr (CLLong) -> CLLong -> Ptr (CTHLongStorage) -> Ptr (Ptr (CLLong)) -> Ptr (Ptr (CLLong)) -> Ptr (CChar) -> CInt -> IO (CInt))