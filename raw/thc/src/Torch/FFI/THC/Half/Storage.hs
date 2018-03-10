{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCHalfStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO (Ptr CTHHalf)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCHalfStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCHalfStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CTHHalf -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCHalfStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> IO CTHHalf

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CTHHalf -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaHalfStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCHalfStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CTHHalf -> CPtrdiff -> IO (Ptr CTHCudaHalfStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCHalfStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCHalfStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCHalfStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCHalfStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCHalfStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCHalfStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CTHHalf -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCHalfStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCHalfStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO (Ptr CTHHalf))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCHalfStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCHalfStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CTHHalf -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCHalfStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> IO CTHHalf)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CTHHalf -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaHalfStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCHalfStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CTHHalf -> CPtrdiff -> IO (Ptr CTHCudaHalfStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCHalfStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCHalfStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCHalfStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCHalfStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCHalfStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCHalfStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CTHHalf -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCHalfStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> IO CInt)