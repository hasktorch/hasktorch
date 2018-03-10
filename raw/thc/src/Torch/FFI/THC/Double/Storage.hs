{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCDoubleStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO (Ptr CDouble)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCDoubleStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCDoubleStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> CDouble -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCDoubleStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> IO CDouble

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CDouble -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaDoubleStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CDouble -> CPtrdiff -> IO (Ptr CTHCudaDoubleStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCDoubleStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCDoubleStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCDoubleStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCDoubleStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCDoubleStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCDoubleStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CDouble -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCDoubleStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCDoubleStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO (Ptr CDouble))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCDoubleStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> CDouble -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCDoubleStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> IO CDouble)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CDouble -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaDoubleStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CDouble -> CPtrdiff -> IO (Ptr CTHCudaDoubleStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> CDouble -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCDoubleStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> IO CInt)