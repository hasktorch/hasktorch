{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.Storage where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCudaCharStorage_data"
  c_data :: Ptr C'THCState -> Ptr C'THCCharStorage -> IO (Ptr CChar)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCudaCharStorage_size"
  c_size :: Ptr C'THCState -> Ptr C'THCCharStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCudaCharStorage_set"
  c_set :: Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> CChar -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCudaCharStorage_get"
  c_get :: Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> IO CChar

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCCharStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithSize"
  c_newWithSize :: Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCCharStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithSize1"
  c_newWithSize1 :: Ptr C'THCState -> CChar -> IO (Ptr C'THCCharStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithSize2"
  c_newWithSize2 :: Ptr C'THCState -> CChar -> CChar -> IO (Ptr C'THCCharStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithSize3"
  c_newWithSize3 :: Ptr C'THCState -> CChar -> CChar -> CChar -> IO (Ptr C'THCCharStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithSize4"
  c_newWithSize4 :: Ptr C'THCState -> CChar -> CChar -> CChar -> CChar -> IO (Ptr C'THCCharStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithMapping"
  c_newWithMapping :: Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCCharStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCudaCharStorage_newWithData"
  c_newWithData :: Ptr C'THCState -> Ptr CChar -> CPtrdiff -> IO (Ptr C'THCCharStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaCharStorage_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaCharStorage_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCudaCharStorage_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCCharStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCudaCharStorage_free"
  c_free :: Ptr C'THCState -> Ptr C'THCCharStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCudaCharStorage_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCudaCharStorage_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCudaCharStorage_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCCharStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCudaCharStorage_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> IO (Ptr CChar))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCudaCharStorage_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> CChar -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCudaCharStorage_get"
  p_get :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> IO CChar)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCCharStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCCharStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr C'THCState -> CChar -> IO (Ptr C'THCCharStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr C'THCState -> CChar -> CChar -> IO (Ptr C'THCCharStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CChar -> IO (Ptr C'THCCharStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CChar -> CChar -> IO (Ptr C'THCCharStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCCharStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaCharStorage_newWithData"
  p_newWithData :: FunPtr (Ptr C'THCState -> Ptr CChar -> CPtrdiff -> IO (Ptr C'THCCharStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCudaCharStorage_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> CChar -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCudaCharStorage_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCCharStorage -> IO CInt)