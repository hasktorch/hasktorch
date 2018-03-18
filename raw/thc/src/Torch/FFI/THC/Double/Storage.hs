{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCDoubleStorage_data"
  c_data :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO (Ptr CDouble)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCDoubleStorage_size"
  c_size :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCDoubleStorage_set"
  c_set :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CDouble -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCDoubleStorage_get"
  c_get :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> IO CDouble

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize"
  c_newWithSize :: Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize1"
  c_newWithSize1 :: Ptr C'THCState -> CDouble -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize2"
  c_newWithSize2 :: Ptr C'THCState -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize3"
  c_newWithSize3 :: Ptr C'THCState -> CDouble -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithSize4"
  c_newWithSize4 :: Ptr C'THCState -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithMapping"
  c_newWithMapping :: Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCDoubleStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCDoubleStorage_newWithData"
  c_newWithData :: Ptr C'THCState -> Ptr CDouble -> CPtrdiff -> IO (Ptr C'THCDoubleStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCDoubleStorage_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCDoubleStorage_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCDoubleStorage_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCDoubleStorage_free"
  c_free :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCDoubleStorage_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCDoubleStorage_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CDouble -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCDoubleStorage_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCDoubleStorage_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO (Ptr CDouble))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCDoubleStorage_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CDouble -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCDoubleStorage_get"
  p_get :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> IO CDouble)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr C'THCState -> CDouble -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr C'THCState -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr C'THCState -> CDouble -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr C'THCState -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCDoubleStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCDoubleStorage_newWithData"
  p_newWithData :: FunPtr (Ptr C'THCState -> Ptr CDouble -> CPtrdiff -> IO (Ptr C'THCDoubleStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCDoubleStorage_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CDouble -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCDoubleStorage_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> IO CInt)