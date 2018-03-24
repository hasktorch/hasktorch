{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.Storage where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_data :   -> real *
foreign import ccall "THStorage.h THFloatStorage_data"
  c_data_ :: Ptr C'THFloatStorage -> IO (Ptr CFloat)

-- | alias of c_data_ with unused argument (for CTHState) to unify backpack signatures.
c_data :: Ptr C'THState -> Ptr C'THFloatStorage -> IO (Ptr CFloat)
c_data = const c_data_

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h THFloatStorage_size"
  c_size_ :: Ptr C'THFloatStorage -> IO CPtrdiff

-- | alias of c_size_ with unused argument (for CTHState) to unify backpack signatures.
c_size :: Ptr C'THState -> Ptr C'THFloatStorage -> IO CPtrdiff
c_size = const c_size_

-- | c_set :     -> void
foreign import ccall "THStorage.h THFloatStorage_set"
  c_set_ :: Ptr C'THFloatStorage -> CPtrdiff -> CFloat -> IO ()

-- | alias of c_set_ with unused argument (for CTHState) to unify backpack signatures.
c_set :: Ptr C'THState -> Ptr C'THFloatStorage -> CPtrdiff -> CFloat -> IO ()
c_set = const c_set_

-- | c_get :    -> real
foreign import ccall "THStorage.h THFloatStorage_get"
  c_get_ :: Ptr C'THFloatStorage -> CPtrdiff -> IO CFloat

-- | alias of c_get_ with unused argument (for CTHState) to unify backpack signatures.
c_get :: Ptr C'THState -> Ptr C'THFloatStorage -> CPtrdiff -> IO CFloat
c_get = const c_get_

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_new"
  c_new_ :: IO (Ptr C'THFloatStorage)

-- | alias of c_new_ with unused argument (for CTHState) to unify backpack signatures.
c_new :: Ptr C'THState -> IO (Ptr C'THFloatStorage)
c_new = const c_new_

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize"
  c_newWithSize_ :: CPtrdiff -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize :: Ptr C'THState -> CPtrdiff -> IO (Ptr C'THFloatStorage)
c_newWithSize = const c_newWithSize_

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize1"
  c_newWithSize1_ :: CFloat -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithSize1_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize1 :: Ptr C'THState -> CFloat -> IO (Ptr C'THFloatStorage)
c_newWithSize1 = const c_newWithSize1_

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize2"
  c_newWithSize2_ :: CFloat -> CFloat -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithSize2_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize2 :: Ptr C'THState -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage)
c_newWithSize2 = const c_newWithSize2_

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize3"
  c_newWithSize3_ :: CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithSize3_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize3 :: Ptr C'THState -> CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage)
c_newWithSize3 = const c_newWithSize3_

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize4"
  c_newWithSize4_ :: CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithSize4_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize4 :: Ptr C'THState -> CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage)
c_newWithSize4 = const c_newWithSize4_

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithMapping"
  c_newWithMapping_ :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithMapping_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithMapping :: Ptr C'THState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THFloatStorage)
c_newWithMapping = const c_newWithMapping_

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithData"
  c_newWithData_ :: Ptr CFloat -> CPtrdiff -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithData_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithData :: Ptr C'THState -> Ptr CFloat -> CPtrdiff -> IO (Ptr C'THFloatStorage)
c_newWithData = const c_newWithData_

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithAllocator"
  c_newWithAllocator_ :: CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithAllocator :: Ptr C'THState -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage)
c_newWithAllocator = const c_newWithAllocator_

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithDataAndAllocator"
  c_newWithDataAndAllocator_ :: Ptr CFloat -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage)

-- | alias of c_newWithDataAndAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithDataAndAllocator :: Ptr C'THState -> Ptr CFloat -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage)
c_newWithDataAndAllocator = const c_newWithDataAndAllocator_

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_setFlag"
  c_setFlag_ :: Ptr C'THFloatStorage -> CChar -> IO ()

-- | alias of c_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_setFlag :: Ptr C'THState -> Ptr C'THFloatStorage -> CChar -> IO ()
c_setFlag = const c_setFlag_

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_clearFlag"
  c_clearFlag_ :: Ptr C'THFloatStorage -> CChar -> IO ()

-- | alias of c_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_clearFlag :: Ptr C'THState -> Ptr C'THFloatStorage -> CChar -> IO ()
c_clearFlag = const c_clearFlag_

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h THFloatStorage_retain"
  c_retain_ :: Ptr C'THFloatStorage -> IO ()

-- | alias of c_retain_ with unused argument (for CTHState) to unify backpack signatures.
c_retain :: Ptr C'THState -> Ptr C'THFloatStorage -> IO ()
c_retain = const c_retain_

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h THFloatStorage_swap"
  c_swap_ :: Ptr C'THFloatStorage -> Ptr C'THFloatStorage -> IO ()

-- | alias of c_swap_ with unused argument (for CTHState) to unify backpack signatures.
c_swap :: Ptr C'THState -> Ptr C'THFloatStorage -> Ptr C'THFloatStorage -> IO ()
c_swap = const c_swap_

-- | c_free :  storage -> void
foreign import ccall "THStorage.h THFloatStorage_free"
  c_free_ :: Ptr C'THFloatStorage -> IO ()

-- | alias of c_free_ with unused argument (for CTHState) to unify backpack signatures.
c_free :: Ptr C'THState -> Ptr C'THFloatStorage -> IO ()
c_free = const c_free_

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h THFloatStorage_resize"
  c_resize_ :: Ptr C'THFloatStorage -> CPtrdiff -> IO ()

-- | alias of c_resize_ with unused argument (for CTHState) to unify backpack signatures.
c_resize :: Ptr C'THState -> Ptr C'THFloatStorage -> CPtrdiff -> IO ()
c_resize = const c_resize_

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h THFloatStorage_fill"
  c_fill_ :: Ptr C'THFloatStorage -> CFloat -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr C'THFloatStorage -> CFloat -> IO ()
c_fill = const c_fill_

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THFloatStorage_data"
  p_data :: FunPtr (Ptr C'THFloatStorage -> IO (Ptr CFloat))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THFloatStorage_size"
  p_size :: FunPtr (Ptr C'THFloatStorage -> IO CPtrdiff)

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THFloatStorage_set"
  p_set :: FunPtr (Ptr C'THFloatStorage -> CPtrdiff -> CFloat -> IO ())

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THFloatStorage_get"
  p_get :: FunPtr (Ptr C'THFloatStorage -> CPtrdiff -> IO CFloat)

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_new"
  p_new :: FunPtr (IO (Ptr C'THFloatStorage))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr C'THFloatStorage))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (CFloat -> IO (Ptr C'THFloatStorage))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (CFloat -> CFloat -> IO (Ptr C'THFloatStorage))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr C'THFloatStorage))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THFloatStorage))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CFloat -> CPtrdiff -> IO (Ptr C'THFloatStorage))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr CFloat -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THFloatStorage))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THFloatStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THFloatStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THFloatStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THFloatStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THFloatStorage_retain"
  p_retain :: FunPtr (Ptr C'THFloatStorage -> IO ())

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THFloatStorage_swap"
  p_swap :: FunPtr (Ptr C'THFloatStorage -> Ptr C'THFloatStorage -> IO ())

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THFloatStorage_free"
  p_free :: FunPtr (Ptr C'THFloatStorage -> IO ())

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THFloatStorage_resize"
  p_resize :: FunPtr (Ptr C'THFloatStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THFloatStorage_fill"
  p_fill :: FunPtr (Ptr C'THFloatStorage -> CFloat -> IO ())