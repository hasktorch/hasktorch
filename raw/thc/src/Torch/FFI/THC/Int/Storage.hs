{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.Storage where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCudaIntStorage_data"
  c_data :: Ptr C'THCState -> Ptr C'THCIntStorage -> IO (Ptr CInt)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCudaIntStorage_size"
  c_size :: Ptr C'THCState -> Ptr C'THCIntStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCudaIntStorage_set"
  c_set :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CInt -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCudaIntStorage_get"
  c_get :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> IO CInt

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCIntStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithSize"
  c_newWithSize :: Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCIntStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithSize1"
  c_newWithSize1 :: Ptr C'THCState -> CInt -> IO (Ptr C'THCIntStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithSize2"
  c_newWithSize2 :: Ptr C'THCState -> CInt -> CInt -> IO (Ptr C'THCIntStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithSize3"
  c_newWithSize3 :: Ptr C'THCState -> CInt -> CInt -> CInt -> IO (Ptr C'THCIntStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithSize4"
  c_newWithSize4 :: Ptr C'THCState -> CInt -> CInt -> CInt -> CInt -> IO (Ptr C'THCIntStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithMapping"
  c_newWithMapping :: Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCIntStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCudaIntStorage_newWithData"
  c_newWithData :: Ptr C'THCState -> Ptr CInt -> CPtrdiff -> IO (Ptr C'THCIntStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaIntStorage_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCIntStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaIntStorage_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCIntStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCudaIntStorage_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCIntStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCudaIntStorage_free"
  c_free :: Ptr C'THCState -> Ptr C'THCIntStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCudaIntStorage_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCudaIntStorage_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCIntStorage -> CInt -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCudaIntStorage_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCIntStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCudaIntStorage_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> IO (Ptr CInt))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCudaIntStorage_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CInt -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCudaIntStorage_get"
  p_get :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> IO CInt)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCIntStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCIntStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr C'THCState -> CInt -> IO (Ptr C'THCIntStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr C'THCState -> CInt -> CInt -> IO (Ptr C'THCIntStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr C'THCState -> CInt -> CInt -> CInt -> IO (Ptr C'THCIntStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr C'THCState -> CInt -> CInt -> CInt -> CInt -> IO (Ptr C'THCIntStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCIntStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaIntStorage_newWithData"
  p_newWithData :: FunPtr (Ptr C'THCState -> Ptr CInt -> CPtrdiff -> IO (Ptr C'THCIntStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCudaIntStorage_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CInt -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCudaIntStorage_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> IO CInt)