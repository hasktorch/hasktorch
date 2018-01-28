{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.Raw.Storage
  ( THStorage(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THByteStorage as S
import qualified THDoubleStorage as S
import qualified THFloatStorage as S
import qualified THIntStorage as S
import qualified THLongStorage as S
import qualified THShortStorage as S

-- CTHDoubleStorage -> CDouble
class THStorage t tt where
  c_data :: Ptr t -> IO (Ptr tt)
  c_size :: Ptr t -> CPtrdiff
  -- c_elementSize :: CSize
  c_set :: Ptr t -> CPtrdiff -> tt -> IO ()
  c_get :: Ptr t -> CPtrdiff -> (HaskReal t)
  c_new :: IO (Ptr t)
  c_newWithSize :: CPtrdiff -> IO (Ptr t)
  c_newWithSize1 :: tt -> IO (Ptr t)
  c_newWithSize2 :: tt -> tt -> IO (Ptr t)
  c_newWithSize3 :: tt -> tt -> tt -> IO (Ptr t)
  c_newWithSize4 :: tt -> tt -> tt -> tt -> IO (Ptr t)
  c_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr t)
  c_newWithData :: Ptr tt -> CPtrdiff -> IO (Ptr t)
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t)
  c_newWithDataAndAllocator :: Ptr tt -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t)
  c_setFlag :: Ptr t -> CChar -> IO ()
  c_clearFlag :: Ptr t -> CChar -> IO ()
  c_retain :: Ptr t -> IO ()
  c_swap :: Ptr t -> Ptr t -> IO ()
  c_free :: Ptr t -> IO ()
  c_resize :: Ptr t -> CPtrdiff -> IO ()
  c_fill :: Ptr t -> tt -> IO ()
  p_data :: FunPtr (Ptr t -> IO (Ptr tt))
  p_size :: FunPtr (Ptr t -> CPtrdiff)
  -- p_elementSize :: FunPtr CSize
  p_set :: FunPtr (Ptr t -> CPtrdiff -> tt -> IO ())
  p_get :: FunPtr (Ptr t -> CPtrdiff -> tt)
  p_new :: FunPtr (IO (Ptr t))
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr t))
  p_newWithSize1 :: FunPtr (tt -> IO (Ptr t))
  p_newWithSize2 :: FunPtr (tt -> tt -> IO (Ptr t))
  p_newWithSize3 :: FunPtr (tt -> tt -> tt -> IO (Ptr t))
  p_newWithSize4 :: FunPtr (tt -> tt -> tt -> tt -> IO (Ptr t))
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr t))
  p_newWithData :: FunPtr (Ptr tt -> CPtrdiff -> IO (Ptr t))
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t))
  p_newWithDataAndAllocator :: FunPtr (Ptr tt -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t))
  p_setFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_clearFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_retain :: FunPtr (Ptr t -> IO ())
  p_swap :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_free :: FunPtr (Ptr t -> IO ())
  p_resize :: FunPtr (Ptr t -> CPtrdiff -> IO ())
  p_fill :: FunPtr (Ptr t -> tt -> IO ())

-- TODO: Complete the implementations
instance THStorage CTHByteStorage CChar where
  c_get = S.c_THByteStorage_get
  c_data = S.c_THByteStorage_data
  c_size = S.c_THByteStorage_size
  -- c_el
  c_set = S.c_THByteStorage_set
  c_new = S.c_THByteStorage_new
  c_newWithSize = S.c_THByteStorage_newWithSize
  c_newWithSize1 = S.c_THByteStorage_newWithSize1
  c_newWithSize2 = S.c_THByteStorage_newWithSize2
  c_newWithSize3 = S.c_THByteStorage_newWithSize3
  c_newWithSize4 = S.c_THByteStorage_newWithSize4
  c_newWithMapping = S.c_THByteStorage_newWithMapping
  c_newWithData = S.c_THByteStorage_newWithData
  c_newWithAllocator = S.c_THByteStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THByteStorage_newWithDataAndAllocator
  c_setFlag = S.c_THByteStorage_setFlag
  c_clearFlag = S.c_THByteStorage_clearFlag
  c_retain = S.c_THByteStorage_retain
  c_swap = S.c_THByteStorage_swap
  c_free = S.c_THByteStorage_free
  c_resize = S.c_THByteStorage_resize
  c_fill = S.c_THByteStorage_fill
  p_data = S.p_THByteStorage_data
  p_size = S.p_THByteStorage_size
  -- p_el
  p_set = S.p_THByteStorage_set
  p_get = S.p_THByteStorage_get
  p_new = S.p_THByteStorage_new
  p_newWithSize = S.p_THByteStorage_newWithSize
  p_newWithSize1 = S.p_THByteStorage_newWithSize1
  p_newWithSize2 = S.p_THByteStorage_newWithSize2
  p_newWithSize3 = S.p_THByteStorage_newWithSize3
  p_newWithSize4 = S.p_THByteStorage_newWithSize4
  p_newWithMapping = S.p_THByteStorage_newWithMapping
  p_newWithData = S.p_THByteStorage_newWithData
  p_newWithAllocator = S.p_THByteStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THByteStorage_newWithDataAndAllocator
  p_setFlag = S.p_THByteStorage_setFlag
  p_clearFlag = S.p_THByteStorage_clearFlag
  p_retain = S.p_THByteStorage_retain
  p_swap = S.p_THByteStorage_swap
  p_free = S.p_THByteStorage_free
  p_resize = S.p_THByteStorage_resize
  p_fill = S.p_THByteStorage_fill

instance THStorage CTHDoubleStorage CDouble where
  c_get = S.c_THDoubleStorage_get
  c_data = S.c_THDoubleStorage_data
  c_size = S.c_THDoubleStorage_size
  -- c_el
  c_set = S.c_THDoubleStorage_set
  c_new = S.c_THDoubleStorage_new
  c_newWithSize = S.c_THDoubleStorage_newWithSize
  c_newWithSize1 = S.c_THDoubleStorage_newWithSize1
  c_newWithSize2 = S.c_THDoubleStorage_newWithSize2
  c_newWithSize3 = S.c_THDoubleStorage_newWithSize3
  c_newWithSize4 = S.c_THDoubleStorage_newWithSize4
  c_newWithMapping = S.c_THDoubleStorage_newWithMapping
  c_newWithData = S.c_THDoubleStorage_newWithData
  c_newWithAllocator = S.c_THDoubleStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THDoubleStorage_newWithDataAndAllocator
  c_setFlag = S.c_THDoubleStorage_setFlag
  c_clearFlag = S.c_THDoubleStorage_clearFlag
  c_retain = S.c_THDoubleStorage_retain
  c_swap = S.c_THDoubleStorage_swap
  c_free = S.c_THDoubleStorage_free
  c_resize = S.c_THDoubleStorage_resize
  c_fill = S.c_THDoubleStorage_fill
  p_data = S.p_THDoubleStorage_data
  p_size = S.p_THDoubleStorage_size
  -- p_el
  p_set = S.p_THDoubleStorage_set
  p_get = S.p_THDoubleStorage_get
  p_new = S.p_THDoubleStorage_new
  p_newWithSize = S.p_THDoubleStorage_newWithSize
  p_newWithSize1 = S.p_THDoubleStorage_newWithSize1
  p_newWithSize2 = S.p_THDoubleStorage_newWithSize2
  p_newWithSize3 = S.p_THDoubleStorage_newWithSize3
  p_newWithSize4 = S.p_THDoubleStorage_newWithSize4
  p_newWithMapping = S.p_THDoubleStorage_newWithMapping
  p_newWithData = S.p_THDoubleStorage_newWithData
  p_newWithAllocator = S.p_THDoubleStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THDoubleStorage_newWithDataAndAllocator
  p_setFlag = S.p_THDoubleStorage_setFlag
  p_clearFlag = S.p_THDoubleStorage_clearFlag
  p_retain = S.p_THDoubleStorage_retain
  p_swap = S.p_THDoubleStorage_swap
  p_free = S.p_THDoubleStorage_free
  p_resize = S.p_THDoubleStorage_resize
  p_fill = S.p_THDoubleStorage_fill

instance THStorage CTHFloatStorage CFloat where
  c_get = S.c_THFloatStorage_get
  c_data = S.c_THFloatStorage_data
  c_size = S.c_THFloatStorage_size
  -- c_el
  c_set = S.c_THFloatStorage_set
  c_new = S.c_THFloatStorage_new
  c_newWithSize = S.c_THFloatStorage_newWithSize
  c_newWithSize1 = S.c_THFloatStorage_newWithSize1
  c_newWithSize2 = S.c_THFloatStorage_newWithSize2
  c_newWithSize3 = S.c_THFloatStorage_newWithSize3
  c_newWithSize4 = S.c_THFloatStorage_newWithSize4
  c_newWithMapping = S.c_THFloatStorage_newWithMapping
  c_newWithData = S.c_THFloatStorage_newWithData
  c_newWithAllocator = S.c_THFloatStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THFloatStorage_newWithDataAndAllocator
  c_setFlag = S.c_THFloatStorage_setFlag
  c_clearFlag = S.c_THFloatStorage_clearFlag
  c_retain = S.c_THFloatStorage_retain
  c_swap = S.c_THFloatStorage_swap
  c_free = S.c_THFloatStorage_free
  c_resize = S.c_THFloatStorage_resize
  c_fill = S.c_THFloatStorage_fill
  p_data = S.p_THFloatStorage_data
  p_size = S.p_THFloatStorage_size
  -- p_el
  p_set = S.p_THFloatStorage_set
  p_get = S.p_THFloatStorage_get
  p_new = S.p_THFloatStorage_new
  p_newWithSize = S.p_THFloatStorage_newWithSize
  p_newWithSize1 = S.p_THFloatStorage_newWithSize1
  p_newWithSize2 = S.p_THFloatStorage_newWithSize2
  p_newWithSize3 = S.p_THFloatStorage_newWithSize3
  p_newWithSize4 = S.p_THFloatStorage_newWithSize4
  p_newWithMapping = S.p_THFloatStorage_newWithMapping
  p_newWithData = S.p_THFloatStorage_newWithData
  p_newWithAllocator = S.p_THFloatStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THFloatStorage_newWithDataAndAllocator
  p_setFlag = S.p_THFloatStorage_setFlag
  p_clearFlag = S.p_THFloatStorage_clearFlag
  p_retain = S.p_THFloatStorage_retain
  p_swap = S.p_THFloatStorage_swap
  p_free = S.p_THFloatStorage_free
  p_resize = S.p_THFloatStorage_resize
  p_fill = S.p_THFloatStorage_fill

instance THStorage CTHIntStorage CInt where
  c_get = S.c_THIntStorage_get
  c_data = S.c_THIntStorage_data
  c_size = S.c_THIntStorage_size
  -- c_el
  c_set = S.c_THIntStorage_set
  c_new = S.c_THIntStorage_new
  c_newWithSize = S.c_THIntStorage_newWithSize
  c_newWithSize1 = S.c_THIntStorage_newWithSize1
  c_newWithSize2 = S.c_THIntStorage_newWithSize2
  c_newWithSize3 = S.c_THIntStorage_newWithSize3
  c_newWithSize4 = S.c_THIntStorage_newWithSize4
  c_newWithMapping = S.c_THIntStorage_newWithMapping
  c_newWithData = S.c_THIntStorage_newWithData
  c_newWithAllocator = S.c_THIntStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THIntStorage_newWithDataAndAllocator
  c_setFlag = S.c_THIntStorage_setFlag
  c_clearFlag = S.c_THIntStorage_clearFlag
  c_retain = S.c_THIntStorage_retain
  c_swap = S.c_THIntStorage_swap
  c_free = S.c_THIntStorage_free
  c_resize = S.c_THIntStorage_resize
  c_fill = S.c_THIntStorage_fill
  p_data = S.p_THIntStorage_data
  p_size = S.p_THIntStorage_size
  -- p_el
  p_set = S.p_THIntStorage_set
  p_get = S.p_THIntStorage_get
  p_new = S.p_THIntStorage_new
  p_newWithSize = S.p_THIntStorage_newWithSize
  p_newWithSize1 = S.p_THIntStorage_newWithSize1
  p_newWithSize2 = S.p_THIntStorage_newWithSize2
  p_newWithSize3 = S.p_THIntStorage_newWithSize3
  p_newWithSize4 = S.p_THIntStorage_newWithSize4
  p_newWithMapping = S.p_THIntStorage_newWithMapping
  p_newWithData = S.p_THIntStorage_newWithData
  p_newWithAllocator = S.p_THIntStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THIntStorage_newWithDataAndAllocator
  p_setFlag = S.p_THIntStorage_setFlag
  p_clearFlag = S.p_THIntStorage_clearFlag
  p_retain = S.p_THIntStorage_retain
  p_swap = S.p_THIntStorage_swap
  p_free = S.p_THIntStorage_free
  p_resize = S.p_THIntStorage_resize
  p_fill = S.p_THIntStorage_fill

instance THStorage CTHLongStorage CLong where
  c_get = S.c_THLongStorage_get
  c_data = S.c_THLongStorage_data
  c_size = S.c_THLongStorage_size
  -- c_el
  c_set = S.c_THLongStorage_set
  c_new = S.c_THLongStorage_new
  c_newWithSize = S.c_THLongStorage_newWithSize
  c_newWithSize1 = S.c_THLongStorage_newWithSize1
  c_newWithSize2 = S.c_THLongStorage_newWithSize2
  c_newWithSize3 = S.c_THLongStorage_newWithSize3
  c_newWithSize4 = S.c_THLongStorage_newWithSize4
  c_newWithMapping = S.c_THLongStorage_newWithMapping
  c_newWithData = S.c_THLongStorage_newWithData
  c_newWithAllocator = S.c_THLongStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THLongStorage_newWithDataAndAllocator
  c_setFlag = S.c_THLongStorage_setFlag
  c_clearFlag = S.c_THLongStorage_clearFlag
  c_retain = S.c_THLongStorage_retain
  c_swap = S.c_THLongStorage_swap
  c_free = S.c_THLongStorage_free
  c_resize = S.c_THLongStorage_resize
  c_fill = S.c_THLongStorage_fill
  p_data = S.p_THLongStorage_data
  p_size = S.p_THLongStorage_size
  -- p_el
  p_set = S.p_THLongStorage_set
  p_get = S.p_THLongStorage_get
  p_new = S.p_THLongStorage_new
  p_newWithSize = S.p_THLongStorage_newWithSize
  p_newWithSize1 = S.p_THLongStorage_newWithSize1
  p_newWithSize2 = S.p_THLongStorage_newWithSize2
  p_newWithSize3 = S.p_THLongStorage_newWithSize3
  p_newWithSize4 = S.p_THLongStorage_newWithSize4
  p_newWithMapping = S.p_THLongStorage_newWithMapping
  p_newWithData = S.p_THLongStorage_newWithData
  p_newWithAllocator = S.p_THLongStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THLongStorage_newWithDataAndAllocator
  p_setFlag = S.p_THLongStorage_setFlag
  p_clearFlag = S.p_THLongStorage_clearFlag
  p_retain = S.p_THLongStorage_retain
  p_swap = S.p_THLongStorage_swap
  p_free = S.p_THLongStorage_free
  p_resize = S.p_THLongStorage_resize
  p_fill = S.p_THLongStorage_fill

instance THStorage CTHShortStorage CShort where
  c_get = S.c_THShortStorage_get
  c_data = S.c_THShortStorage_data
  c_size = S.c_THShortStorage_size
  -- c_el
  c_set = S.c_THShortStorage_set
  c_new = S.c_THShortStorage_new
  c_newWithSize = S.c_THShortStorage_newWithSize
  c_newWithSize1 = S.c_THShortStorage_newWithSize1
  c_newWithSize2 = S.c_THShortStorage_newWithSize2
  c_newWithSize3 = S.c_THShortStorage_newWithSize3
  c_newWithSize4 = S.c_THShortStorage_newWithSize4
  c_newWithMapping = S.c_THShortStorage_newWithMapping
  c_newWithData = S.c_THShortStorage_newWithData
  c_newWithAllocator = S.c_THShortStorage_newWithAllocator
  c_newWithDataAndAllocator = S.c_THShortStorage_newWithDataAndAllocator
  c_setFlag = S.c_THShortStorage_setFlag
  c_clearFlag = S.c_THShortStorage_clearFlag
  c_retain = S.c_THShortStorage_retain
  c_swap = S.c_THShortStorage_swap
  c_free = S.c_THShortStorage_free
  c_resize = S.c_THShortStorage_resize
  c_fill = S.c_THShortStorage_fill
  p_data = S.p_THShortStorage_data
  p_size = S.p_THShortStorage_size
  -- p_el
  p_set = S.p_THShortStorage_set
  p_get = S.p_THShortStorage_get
  p_new = S.p_THShortStorage_new
  p_newWithSize = S.p_THShortStorage_newWithSize
  p_newWithSize1 = S.p_THShortStorage_newWithSize1
  p_newWithSize2 = S.p_THShortStorage_newWithSize2
  p_newWithSize3 = S.p_THShortStorage_newWithSize3
  p_newWithSize4 = S.p_THShortStorage_newWithSize4
  p_newWithMapping = S.p_THShortStorage_newWithMapping
  p_newWithData = S.p_THShortStorage_newWithData
  p_newWithAllocator = S.p_THShortStorage_newWithAllocator
  p_newWithDataAndAllocator = S.p_THShortStorage_newWithDataAndAllocator
  p_setFlag = S.p_THShortStorage_setFlag
  p_clearFlag = S.p_THShortStorage_clearFlag
  p_retain = S.p_THShortStorage_retain
  p_swap = S.p_THShortStorage_swap
  p_free = S.p_THShortStorage_free
  p_resize = S.p_THShortStorage_resize
  p_fill = S.p_THShortStorage_fill

