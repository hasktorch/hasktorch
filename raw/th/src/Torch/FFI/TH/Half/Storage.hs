{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.Storage where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :   -> real *
foreign import ccall "THStorage.h THHalfStorage_data"
  c_data_ :: Ptr C'THHalfStorage -> IO (Ptr CTHHalf)

-- | alias of c_data_ with unused argument (for CTHState) to unify backpack signatures.
c_data = const c_data_

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h THHalfStorage_size"
  c_size_ :: Ptr C'THHalfStorage -> IO CPtrdiff

-- | alias of c_size_ with unused argument (for CTHState) to unify backpack signatures.
c_size = const c_size_

-- | c_set :     -> void
foreign import ccall "THStorage.h THHalfStorage_set"
  c_set_ :: Ptr C'THHalfStorage -> CPtrdiff -> CTHHalf -> IO ()

-- | alias of c_set_ with unused argument (for CTHState) to unify backpack signatures.
c_set = const c_set_

-- | c_get :    -> real
foreign import ccall "THStorage.h THHalfStorage_get"
  c_get_ :: Ptr C'THHalfStorage -> CPtrdiff -> IO CTHHalf

-- | alias of c_get_ with unused argument (for CTHState) to unify backpack signatures.
c_get = const c_get_

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_new"
  c_new_ :: IO (Ptr C'THHalfStorage)

-- | alias of c_new_ with unused argument (for CTHState) to unify backpack signatures.
c_new = const c_new_

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize"
  c_newWithSize_ :: CPtrdiff -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize = const c_newWithSize_

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize1"
  c_newWithSize1_ :: CTHHalf -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithSize1_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize1 = const c_newWithSize1_

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize2"
  c_newWithSize2_ :: CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithSize2_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize2 = const c_newWithSize2_

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize3"
  c_newWithSize3_ :: CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithSize3_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize3 = const c_newWithSize3_

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize4"
  c_newWithSize4_ :: CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithSize4_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize4 = const c_newWithSize4_

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithMapping"
  c_newWithMapping_ :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithMapping_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithMapping = const c_newWithMapping_

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithData"
  c_newWithData_ :: Ptr CTHHalf -> CPtrdiff -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithData_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithData = const c_newWithData_

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithAllocator"
  c_newWithAllocator_ :: CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithAllocator = const c_newWithAllocator_

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithDataAndAllocator"
  c_newWithDataAndAllocator_ :: Ptr CTHHalf -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THHalfStorage)

-- | alias of c_newWithDataAndAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithDataAndAllocator = const c_newWithDataAndAllocator_

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h THHalfStorage_setFlag"
  c_setFlag_ :: Ptr C'THHalfStorage -> CChar -> IO ()

-- | alias of c_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_setFlag = const c_setFlag_

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h THHalfStorage_clearFlag"
  c_clearFlag_ :: Ptr C'THHalfStorage -> CChar -> IO ()

-- | alias of c_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_clearFlag = const c_clearFlag_

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h THHalfStorage_retain"
  c_retain_ :: Ptr C'THHalfStorage -> IO ()

-- | alias of c_retain_ with unused argument (for CTHState) to unify backpack signatures.
c_retain = const c_retain_

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h THHalfStorage_swap"
  c_swap_ :: Ptr C'THHalfStorage -> Ptr C'THHalfStorage -> IO ()

-- | alias of c_swap_ with unused argument (for CTHState) to unify backpack signatures.
c_swap = const c_swap_

-- | c_free :  storage -> void
foreign import ccall "THStorage.h THHalfStorage_free"
  c_free_ :: Ptr C'THHalfStorage -> IO ()

-- | alias of c_free_ with unused argument (for CTHState) to unify backpack signatures.
c_free = const c_free_

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h THHalfStorage_resize"
  c_resize_ :: Ptr C'THHalfStorage -> CPtrdiff -> IO ()

-- | alias of c_resize_ with unused argument (for CTHState) to unify backpack signatures.
c_resize = const c_resize_

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h THHalfStorage_fill"
  c_fill_ :: Ptr C'THHalfStorage -> CTHHalf -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THHalfStorage_data"
  p_data_ :: FunPtr (Ptr C'THHalfStorage -> IO (Ptr CTHHalf))

-- | alias of p_data_ with unused argument (for CTHState) to unify backpack signatures.
p_data = const p_data_

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THHalfStorage_size"
  p_size_ :: FunPtr (Ptr C'THHalfStorage -> IO CPtrdiff)

-- | alias of p_size_ with unused argument (for CTHState) to unify backpack signatures.
p_size = const p_size_

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THHalfStorage_set"
  p_set_ :: FunPtr (Ptr C'THHalfStorage -> CPtrdiff -> CTHHalf -> IO ())

-- | alias of p_set_ with unused argument (for CTHState) to unify backpack signatures.
p_set = const p_set_

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THHalfStorage_get"
  p_get_ :: FunPtr (Ptr C'THHalfStorage -> CPtrdiff -> IO CTHHalf)

-- | alias of p_get_ with unused argument (for CTHState) to unify backpack signatures.
p_get = const p_get_

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_new"
  p_new_ :: FunPtr (IO (Ptr C'THHalfStorage))

-- | alias of p_new_ with unused argument (for CTHState) to unify backpack signatures.
p_new = const p_new_

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize"
  p_newWithSize_ :: FunPtr (CPtrdiff -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize = const p_newWithSize_

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize1"
  p_newWithSize1_ :: FunPtr (CTHHalf -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithSize1_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize1 = const p_newWithSize1_

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize2"
  p_newWithSize2_ :: FunPtr (CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithSize2_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize2 = const p_newWithSize2_

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize3"
  p_newWithSize3_ :: FunPtr (CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithSize3_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize3 = const p_newWithSize3_

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize4"
  p_newWithSize4_ :: FunPtr (CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithSize4_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize4 = const p_newWithSize4_

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithMapping"
  p_newWithMapping_ :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithMapping_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithMapping = const p_newWithMapping_

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithData"
  p_newWithData_ :: FunPtr (Ptr CTHHalf -> CPtrdiff -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithData_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithData = const p_newWithData_

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithAllocator"
  p_newWithAllocator_ :: FunPtr (CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithAllocator_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithAllocator = const p_newWithAllocator_

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithDataAndAllocator"
  p_newWithDataAndAllocator_ :: FunPtr (Ptr CTHHalf -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THHalfStorage))

-- | alias of p_newWithDataAndAllocator_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithDataAndAllocator = const p_newWithDataAndAllocator_

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THHalfStorage_setFlag"
  p_setFlag_ :: FunPtr (Ptr C'THHalfStorage -> CChar -> IO ())

-- | alias of p_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
p_setFlag = const p_setFlag_

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THHalfStorage_clearFlag"
  p_clearFlag_ :: FunPtr (Ptr C'THHalfStorage -> CChar -> IO ())

-- | alias of p_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
p_clearFlag = const p_clearFlag_

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THHalfStorage_retain"
  p_retain_ :: FunPtr (Ptr C'THHalfStorage -> IO ())

-- | alias of p_retain_ with unused argument (for CTHState) to unify backpack signatures.
p_retain = const p_retain_

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THHalfStorage_swap"
  p_swap_ :: FunPtr (Ptr C'THHalfStorage -> Ptr C'THHalfStorage -> IO ())

-- | alias of p_swap_ with unused argument (for CTHState) to unify backpack signatures.
p_swap = const p_swap_

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THHalfStorage_free"
  p_free_ :: FunPtr (Ptr C'THHalfStorage -> IO ())

-- | alias of p_free_ with unused argument (for CTHState) to unify backpack signatures.
p_free = const p_free_

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THHalfStorage_resize"
  p_resize_ :: FunPtr (Ptr C'THHalfStorage -> CPtrdiff -> IO ())

-- | alias of p_resize_ with unused argument (for CTHState) to unify backpack signatures.
p_resize = const p_resize_

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THHalfStorage_fill"
  p_fill_ :: FunPtr (Ptr C'THHalfStorage -> CTHHalf -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_