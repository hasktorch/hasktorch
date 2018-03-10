{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCFloatStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO (Ptr CFloat)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCFloatStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCFloatStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> CFloat -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCFloatStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> IO CFloat

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CFloat -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaFloatStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCFloatStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CFloat -> CPtrdiff -> IO (Ptr CTHCudaFloatStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCFloatStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCFloatStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCFloatStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCFloatStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCFloatStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCFloatStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CFloat -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCFloatStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCFloatStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO (Ptr CFloat))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCFloatStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCFloatStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> CFloat -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCFloatStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> IO CFloat)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CFloat -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaFloatStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCFloatStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CFloat -> CPtrdiff -> IO (Ptr CTHCudaFloatStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCFloatStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCFloatStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCFloatStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCFloatStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCFloatStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCFloatStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> CFloat -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCFloatStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> IO CInt)