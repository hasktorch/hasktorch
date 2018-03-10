{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Storage
  ( c_data
  , c_size
  , c_set
  , c_get
  , c_new
  , c_newWithSize
  , c_newWithSize1
  , c_newWithSize2
  , c_newWithSize3
  , c_newWithSize4
  , c_newWithMapping
  , c_newWithData
  , c_newWithAllocator
  , c_newWithDataAndAllocator
  , c_setFlag
  , c_clearFlag
  , c_retain
  , c_swap
  , c_free
  , c_resize
  , c_fill
  , p_data
  , p_size
  , p_set
  , p_get
  , p_new
  , p_newWithSize
  , p_newWithSize1
  , p_newWithSize2
  , p_newWithSize3
  , p_newWithSize4
  , p_newWithMapping
  , p_newWithData
  , p_newWithAllocator
  , p_newWithDataAndAllocator
  , p_setFlag
  , p_clearFlag
  , p_retain
  , p_swap
  , p_free
  , p_resize
  , p_fill
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :   -> real *
foreign import ccall "THStorage.h THShortStorage_data"
  c_data :: Ptr CTHShortStorage -> IO (Ptr CShort)

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h THShortStorage_size"
  c_size :: Ptr CTHShortStorage -> IO CPtrdiff

-- | c_set :     -> void
foreign import ccall "THStorage.h THShortStorage_set"
  c_set :: Ptr CTHShortStorage -> CPtrdiff -> CShort -> IO ()

-- | c_get :    -> real
foreign import ccall "THStorage.h THShortStorage_get"
  c_get :: Ptr CTHShortStorage -> CPtrdiff -> IO CShort

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h THShortStorage_new"
  c_new :: IO (Ptr CTHShortStorage)

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr CTHShortStorage)

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize1"
  c_newWithSize1 :: CShort -> IO (Ptr CTHShortStorage)

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize2"
  c_newWithSize2 :: CShort -> CShort -> IO (Ptr CTHShortStorage)

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize3"
  c_newWithSize3 :: CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize4"
  c_newWithSize4 :: CShort -> CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithMapping"
  c_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHShortStorage)

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithData"
  c_newWithData :: Ptr CShort -> CPtrdiff -> IO (Ptr CTHShortStorage)

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHShortStorage)

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr CShort -> CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHShortStorage)

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h THShortStorage_setFlag"
  c_setFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h THShortStorage_clearFlag"
  c_clearFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h THShortStorage_retain"
  c_retain :: Ptr CTHShortStorage -> IO ()

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h THShortStorage_swap"
  c_swap :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- | c_free :  storage -> void
foreign import ccall "THStorage.h THShortStorage_free"
  c_free :: Ptr CTHShortStorage -> IO ()

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h THShortStorage_resize"
  c_resize :: Ptr CTHShortStorage -> CPtrdiff -> IO ()

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h THShortStorage_fill"
  c_fill :: Ptr CTHShortStorage -> CShort -> IO ()

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THShortStorage_data"
  p_data :: FunPtr (Ptr CTHShortStorage -> IO (Ptr CShort))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THShortStorage_size"
  p_size :: FunPtr (Ptr CTHShortStorage -> IO CPtrdiff)

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THShortStorage_set"
  p_set :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CShort -> IO ())

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THShortStorage_get"
  p_get :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> IO CShort)

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_new"
  p_new :: FunPtr (IO (Ptr CTHShortStorage))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHShortStorage))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (CShort -> IO (Ptr CTHShortStorage))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (CShort -> CShort -> IO (Ptr CTHShortStorage))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (CShort -> CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHShortStorage))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CShort -> CPtrdiff -> IO (Ptr CTHShortStorage))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHShortStorage))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THShortStorage_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr CShort -> CPtrdiff -> Ptr CTHAllocator -> Ptr () -> IO (Ptr CTHShortStorage))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THShortStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHShortStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THShortStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHShortStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THShortStorage_retain"
  p_retain :: FunPtr (Ptr CTHShortStorage -> IO ())

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THShortStorage_swap"
  p_swap :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THShortStorage_free"
  p_free :: FunPtr (Ptr CTHShortStorage -> IO ())

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THShortStorage_resize"
  p_resize :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THShortStorage_fill"
  p_fill :: FunPtr (Ptr CTHShortStorage -> CShort -> IO ())