{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :   -> real *
foreign import ccall "THStorage.h THLongStorage_data"
  c_data :: Ptr CTHLongStorage -> IO (Ptr CLong)

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h THLongStorage_size"
  c_size :: Ptr CTHLongStorage -> IO CPtrdiff

-- | c_set :     -> void
foreign import ccall "THStorage.h THLongStorage_set"
  c_set :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> IO ()

-- | c_get :    -> real
foreign import ccall "THStorage.h THLongStorage_get"
  c_get :: Ptr CTHLongStorage -> CPtrdiff -> IO CLong

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h THLongStorage_new"
  c_new :: IO (Ptr CTHLongStorage)

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr CTHLongStorage)

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize1"
  c_newWithSize1 :: CLong -> IO (Ptr CTHLongStorage)

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize2"
  c_newWithSize2 :: CLong -> CLong -> IO (Ptr CTHLongStorage)

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize3"
  c_newWithSize3 :: CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize4"
  c_newWithSize4 :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithMapping"
  c_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHLongStorage)

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithData"
  c_newWithData :: Ptr CLong -> CPtrdiff -> IO (Ptr CTHLongStorage)

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHLongStorage)

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr CLong -> CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHLongStorage)

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h THLongStorage_setFlag"
  c_setFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h THLongStorage_clearFlag"
  c_clearFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h THLongStorage_retain"
  c_retain :: Ptr CTHLongStorage -> IO ()

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h THLongStorage_swap"
  c_swap :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- | c_free :  storage -> void
foreign import ccall "THStorage.h THLongStorage_free"
  c_free :: Ptr CTHLongStorage -> IO ()

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h THLongStorage_resize"
  c_resize :: Ptr CTHLongStorage -> CPtrdiff -> IO ()

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h THLongStorage_fill"
  c_fill :: Ptr CTHLongStorage -> CLong -> IO ()

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THLongStorage_data"
  p_data :: FunPtr (Ptr CTHLongStorage -> IO (Ptr CLong))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THLongStorage_size"
  p_size :: FunPtr (Ptr CTHLongStorage -> IO CPtrdiff)

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THLongStorage_set"
  p_set :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> IO ())

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THLongStorage_get"
  p_get :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> IO CLong)

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_new"
  p_new :: FunPtr (IO (Ptr CTHLongStorage))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHLongStorage))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (CLong -> IO (Ptr CTHLongStorage))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (CLong -> CLong -> IO (Ptr CTHLongStorage))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHLongStorage))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CLong -> CPtrdiff -> IO (Ptr CTHLongStorage))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHLongStorage))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THLongStorage_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr CLong -> CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHLongStorage))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THLongStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHLongStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THLongStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHLongStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THLongStorage_retain"
  p_retain :: FunPtr (Ptr CTHLongStorage -> IO ())

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THLongStorage_swap"
  p_swap :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THLongStorage_free"
  p_free :: FunPtr (Ptr CTHLongStorage -> IO ())

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THLongStorage_resize"
  p_resize :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THLongStorage_fill"
  p_fill :: FunPtr (Ptr CTHLongStorage -> CLong -> IO ())