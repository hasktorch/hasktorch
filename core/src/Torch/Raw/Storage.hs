module Torch.Raw.Storage
  ( THStorage(..)
  , module X
  ) where

import Torch.Raw.Internal as X

-- CTHDoubleStorage -> CDouble
class THStorage t where
  c_data :: Ptr t -> IO (Ptr CDouble)
  c_size :: Ptr t -> CPtrdiff
  -- c_elementSize :: CSize
  c_set :: Ptr t -> CPtrdiff -> CDouble -> IO ()
  c_get :: Ptr t -> CPtrdiff -> CDouble
  c_new :: IO (Ptr t)
  c_newWithSize :: CPtrdiff -> IO (Ptr t)
  c_newWithSize1 :: CDouble -> IO (Ptr t)
  c_newWithSize2 :: CDouble -> CDouble -> IO (Ptr t)
  c_newWithSize3 :: CDouble -> CDouble -> CDouble -> IO (Ptr t)
  c_newWithSize4 :: CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr t)
  c_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr t)
  c_newWithData :: Ptr CDouble -> CPtrdiff -> IO (Ptr t)
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t)
  c_newWithDataAndAllocator :: Ptr CDouble -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t)
  c_setFlag :: Ptr t -> CChar -> IO ()
  c_clearFlag :: Ptr t -> CChar -> IO ()
  c_retain :: Ptr t -> IO ()
  c_swap :: Ptr t -> Ptr t -> IO ()
  c_free :: Ptr t -> IO ()
  c_resize :: Ptr t -> CPtrdiff -> IO ()
  c_fill :: Ptr t -> CDouble -> IO ()
  p_data :: FunPtr (Ptr t -> IO (Ptr CDouble))
  p_size :: FunPtr (Ptr t -> CPtrdiff)
  -- p_elementSize :: FunPtr CSize
  p_set :: FunPtr (Ptr t -> CPtrdiff -> CDouble -> IO ())
  p_get :: FunPtr (Ptr t -> CPtrdiff -> CDouble)
  p_new :: FunPtr (IO (Ptr t))
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr t))
  p_newWithSize1 :: FunPtr (CDouble -> IO (Ptr t))
  p_newWithSize2 :: FunPtr (CDouble -> CDouble -> IO (Ptr t))
  p_newWithSize3 :: FunPtr (CDouble -> CDouble -> CDouble -> IO (Ptr t))
  p_newWithSize4 :: FunPtr (CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr t))
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr t))
  p_newWithData :: FunPtr (Ptr CDouble -> CPtrdiff -> IO (Ptr t))
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t))
  p_newWithDataAndAllocator :: FunPtr (Ptr CDouble -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr t))
  p_setFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_clearFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_retain :: FunPtr (Ptr t -> IO ())
  p_swap :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_free :: FunPtr (Ptr t -> IO ())
  p_resize :: FunPtr (Ptr t -> CPtrdiff -> IO ())
  p_fill :: FunPtr (Ptr t -> CDouble -> IO ())
