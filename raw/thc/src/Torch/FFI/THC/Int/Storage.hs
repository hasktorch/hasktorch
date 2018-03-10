{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCIntStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO (Ptr CInt)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCIntStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCIntStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> CInt -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCIntStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> IO CInt

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CInt -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CInt -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CInt -> CInt -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaIntStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCIntStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CInt -> CPtrdiff -> IO (Ptr CTHCudaIntStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCIntStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCIntStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCIntStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCIntStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCIntStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCIntStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CInt -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCIntStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCIntStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO (Ptr CInt))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCIntStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCIntStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> CInt -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCIntStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> IO CInt)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CInt -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CInt -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CInt -> CInt -> CInt -> CInt -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaIntStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCIntStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CInt -> CPtrdiff -> IO (Ptr CTHCudaIntStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCIntStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCIntStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCIntStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCIntStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCIntStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCIntStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> CInt -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCIntStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntStorage -> IO CInt)