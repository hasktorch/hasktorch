{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Storage where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_data :   -> real *
foreign import ccall "THStorage.h THCharStorage_data"
  c_data_ :: Ptr C'THCharStorage -> IO (Ptr CChar)

-- | alias of c_data_ with unused argument (for CTHState) to unify backpack signatures.
c_data :: Ptr C'THState -> Ptr C'THCharStorage -> IO (Ptr CChar)
c_data = const c_data_

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h THCharStorage_size"
  c_size_ :: Ptr C'THCharStorage -> IO CPtrdiff

-- | alias of c_size_ with unused argument (for CTHState) to unify backpack signatures.
c_size :: Ptr C'THState -> Ptr C'THCharStorage -> IO CPtrdiff
c_size = const c_size_

-- | c_set :     -> void
foreign import ccall "THStorage.h THCharStorage_set"
  c_set_ :: Ptr C'THCharStorage -> CPtrdiff -> CChar -> IO ()

-- | alias of c_set_ with unused argument (for CTHState) to unify backpack signatures.
c_set :: Ptr C'THState -> Ptr C'THCharStorage -> CPtrdiff -> CChar -> IO ()
c_set = const c_set_

-- | c_get :    -> real
foreign import ccall "THStorage.h THCharStorage_get"
  c_get_ :: Ptr C'THCharStorage -> CPtrdiff -> IO CChar

-- | alias of c_get_ with unused argument (for CTHState) to unify backpack signatures.
c_get :: Ptr C'THState -> Ptr C'THCharStorage -> CPtrdiff -> IO CChar
c_get = const c_get_

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h THCharStorage_new"
  c_new_ :: IO (Ptr C'THCharStorage)

-- | alias of c_new_ with unused argument (for CTHState) to unify backpack signatures.
c_new :: Ptr C'THState -> IO (Ptr C'THCharStorage)
c_new = const c_new_

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithSize"
  c_newWithSize_ :: CPtrdiff -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize :: Ptr C'THState -> CPtrdiff -> IO (Ptr C'THCharStorage)
c_newWithSize = const c_newWithSize_

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithSize1"
  c_newWithSize1_ :: CChar -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithSize1_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize1 :: Ptr C'THState -> CChar -> IO (Ptr C'THCharStorage)
c_newWithSize1 = const c_newWithSize1_

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithSize2"
  c_newWithSize2_ :: CChar -> CChar -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithSize2_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize2 :: Ptr C'THState -> CChar -> CChar -> IO (Ptr C'THCharStorage)
c_newWithSize2 = const c_newWithSize2_

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithSize3"
  c_newWithSize3_ :: CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithSize3_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize3 :: Ptr C'THState -> CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage)
c_newWithSize3 = const c_newWithSize3_

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithSize4"
  c_newWithSize4_ :: CChar -> CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithSize4_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize4 :: Ptr C'THState -> CChar -> CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage)
c_newWithSize4 = const c_newWithSize4_

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithMapping"
  c_newWithMapping_ :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithMapping_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithMapping :: Ptr C'THState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCharStorage)
c_newWithMapping = const c_newWithMapping_

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithData"
  c_newWithData_ :: Ptr CChar -> CPtrdiff -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithData_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithData :: Ptr C'THState -> Ptr CChar -> CPtrdiff -> IO (Ptr C'THCharStorage)
c_newWithData = const c_newWithData_

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithAllocator"
  c_newWithAllocator_ :: CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithAllocator :: Ptr C'THState -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage)
c_newWithAllocator = const c_newWithAllocator_

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THCharStorage_newWithDataAndAllocator"
  c_newWithDataAndAllocator_ :: Ptr CChar -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage)

-- | alias of c_newWithDataAndAllocator_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithDataAndAllocator :: Ptr C'THState -> Ptr CChar -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage)
c_newWithDataAndAllocator = const c_newWithDataAndAllocator_

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h THCharStorage_setFlag"
  c_setFlag_ :: Ptr C'THCharStorage -> CChar -> IO ()

-- | alias of c_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_setFlag :: Ptr C'THState -> Ptr C'THCharStorage -> CChar -> IO ()
c_setFlag = const c_setFlag_

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h THCharStorage_clearFlag"
  c_clearFlag_ :: Ptr C'THCharStorage -> CChar -> IO ()

-- | alias of c_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_clearFlag :: Ptr C'THState -> Ptr C'THCharStorage -> CChar -> IO ()
c_clearFlag = const c_clearFlag_

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h THCharStorage_retain"
  c_retain_ :: Ptr C'THCharStorage -> IO ()

-- | alias of c_retain_ with unused argument (for CTHState) to unify backpack signatures.
c_retain :: Ptr C'THState -> Ptr C'THCharStorage -> IO ()
c_retain = const c_retain_

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h THCharStorage_swap"
  c_swap_ :: Ptr C'THCharStorage -> Ptr C'THCharStorage -> IO ()

-- | alias of c_swap_ with unused argument (for CTHState) to unify backpack signatures.
c_swap :: Ptr C'THState -> Ptr C'THCharStorage -> Ptr C'THCharStorage -> IO ()
c_swap = const c_swap_

-- | c_free :  storage -> void
foreign import ccall "THStorage.h THCharStorage_free"
  c_free_ :: Ptr C'THCharStorage -> IO ()

-- | alias of c_free_ with unused argument (for CTHState) to unify backpack signatures.
c_free :: Ptr C'THState -> Ptr C'THCharStorage -> IO ()
c_free = const c_free_

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h THCharStorage_resize"
  c_resize_ :: Ptr C'THCharStorage -> CPtrdiff -> IO ()

-- | alias of c_resize_ with unused argument (for CTHState) to unify backpack signatures.
c_resize :: Ptr C'THState -> Ptr C'THCharStorage -> CPtrdiff -> IO ()
c_resize = const c_resize_

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h THCharStorage_fill"
  c_fill_ :: Ptr C'THCharStorage -> CChar -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr C'THCharStorage -> CChar -> IO ()
c_fill = const c_fill_

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THCharStorage_data"
  p_data :: FunPtr (Ptr C'THCharStorage -> IO (Ptr CChar))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THCharStorage_size"
  p_size :: FunPtr (Ptr C'THCharStorage -> IO CPtrdiff)

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THCharStorage_set"
  p_set :: FunPtr (Ptr C'THCharStorage -> CPtrdiff -> CChar -> IO ())

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THCharStorage_get"
  p_get :: FunPtr (Ptr C'THCharStorage -> CPtrdiff -> IO CChar)

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_new"
  p_new :: FunPtr (IO (Ptr C'THCharStorage))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr C'THCharStorage))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (CChar -> IO (Ptr C'THCharStorage))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (CChar -> CChar -> IO (Ptr C'THCharStorage))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (CChar -> CChar -> CChar -> CChar -> IO (Ptr C'THCharStorage))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCharStorage))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CChar -> CPtrdiff -> IO (Ptr C'THCharStorage))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THCharStorage_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr CChar -> CPtrdiff -> Ptr C'THAllocator -> Ptr () -> IO (Ptr C'THCharStorage))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THCharStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCharStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THCharStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCharStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THCharStorage_retain"
  p_retain :: FunPtr (Ptr C'THCharStorage -> IO ())

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THCharStorage_swap"
  p_swap :: FunPtr (Ptr C'THCharStorage -> Ptr C'THCharStorage -> IO ())

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THCharStorage_free"
  p_free :: FunPtr (Ptr C'THCharStorage -> IO ())

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THCharStorage_resize"
  p_resize :: FunPtr (Ptr C'THCharStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THCharStorage_fill"
  p_fill :: FunPtr (Ptr C'THCharStorage -> CChar -> IO ())