{-# LANGUAGE ForeignFunctionInterface #-}

module THStorage (
    c_THLongStorage_sizeDesc,
    c_THLongStorage_newInferSize,
    c_THLongStorage_inferSize2,
    c_THLongStorage_inferSizeN,
    c_THLongStorage_inferExpandGeometry) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongStorage_sizeDesc : size -> THDescBuff
foreign import ccall "THStorage.h THLongStorage_sizeDesc"
  c_THLongStorage_sizeDesc :: Ptr CTHLongStorage -> CTHDescBuff

-- |c_THLongStorage_newInferSize : size nElement -> THLongStorage *
foreign import ccall "THStorage.h THLongStorage_newInferSize"
  c_THLongStorage_newInferSize :: Ptr CTHLongStorage -> CPtrdiff -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_inferSize2 : output sizesA dimsA sizesB dimsB error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSize2"
  c_THLongStorage_inferSize2 :: Ptr CTHLongStorage -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> Ptr CChar -> CInt -> CInt

-- |c_THLongStorage_inferSizeN : output n sizes dims error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferSizeN"
  c_THLongStorage_inferSizeN :: Ptr CTHLongStorage -> CInt -> Ptr (Ptr CLong) -> Ptr CLong -> Ptr CChar -> CInt -> CInt

-- |c_THLongStorage_inferExpandGeometry : tensorSizes tensorStrides tensorDim sizes expandedSizes expandedStrides error_buffer buffer_len -> int
foreign import ccall "THStorage.h THLongStorage_inferExpandGeometry"
  c_THLongStorage_inferExpandGeometry :: Ptr CLong -> Ptr CLong -> CLong -> Ptr CTHLongStorage -> Ptr (Ptr CLong) -> Ptr (Ptr CLong) -> Ptr CChar -> CInt -> CInt