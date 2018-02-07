{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Storage
  ( Storage(..)
  , asStorage
  ) where

import Foreign (Ptr, withForeignPtr, newForeignPtr)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import Torch.Class.Internal (HsReal)
import THTypes

import Torch.Core.Types hiding (HsReal)
import qualified Storage as Sig
import qualified Torch.Class.Storage as Class

asStorage :: Ptr CStorage -> IO Storage
asStorage = fmap Storage . newForeignPtr Sig.p_free

instance Class.IsStorage Storage where
  -- tensordata :: Storage -> IO (Ptr (HsReal Storage))
  -- tensordata s = withForeignPtr (storage s) Sig.c_data

  size :: Storage -> IO CPtrdiff
  size s = withForeignPtr (storage s) (pure . Sig.c_size)

  set :: Storage -> CPtrdiff -> HsReal Storage -> IO ()
  set s pd v = withForeignPtr (storage s) $ \s' -> Sig.c_set s' pd (hs2cReal v)

  get :: Storage -> CPtrdiff -> IO (HsReal Storage)
  get s pd = withForeignPtr (storage s)  $ \s' -> pure . c2hsReal $ Sig.c_get s' pd

  new :: IO Storage
  new = Sig.c_new >>= asStorage

  newWithSize :: CPtrdiff -> IO Storage
  newWithSize pd = Sig.c_newWithSize pd >>= asStorage

  newWithSize1 :: HsReal Storage -> IO Storage
  newWithSize1 a0 = Sig.c_newWithSize1 (hs2cReal a0) >>= asStorage

  newWithSize2 :: HsReal Storage -> HsReal Storage -> IO Storage
  newWithSize2 a0 a1 = Sig.c_newWithSize2 (hs2cReal a0) (hs2cReal a1) >>= asStorage

  newWithSize3 :: HsReal Storage -> HsReal Storage -> HsReal Storage -> IO Storage
  newWithSize3 a0 a1 a2 = Sig.c_newWithSize3 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) >>= asStorage

  newWithSize4 :: HsReal Storage -> HsReal Storage -> HsReal Storage -> HsReal Storage -> IO Storage
  newWithSize4 a0 a1 a2 a3 = Sig.c_newWithSize4 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3) >>= asStorage

  newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO Storage
  newWithMapping pcc pd ci = Sig.c_newWithMapping pcc pd ci >>= asStorage

  -- newWithData :: Ptr (HsReal Storage) -> CPtrdiff -> IO Storage
  -- newWithData pr pd = Sig.c_newWithData pr pd >>= asStorage

  newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO Storage
  newWithAllocator pd calloc whatthehell = Sig.c_newWithAllocator pd calloc whatthehell >>= asStorage

  -- newWithDataAndAllocator :: Ptr (HsReal Storage) -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO Storage
  -- newWithDataAndAllocator pr pd calloc whatthehell = Sig.c_newWithDataAndAllocator pr pd calloc whatthehell >>= asStorage

  setFlag :: Storage -> CChar -> IO ()
  setFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_setFlag s' cc

  clearFlag :: Storage -> CChar -> IO ()
  clearFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_clearFlag s' cc

  retain :: Storage -> IO ()
  retain s = withForeignPtr (storage s) Sig.c_retain

  swap :: Storage -> Storage -> IO ()
  swap s0 s1 =
    withForeignPtr (storage s0) $ \s0' ->
      withForeignPtr (storage s1) $ \s1' ->
        Sig.c_swap s0' s1'

  resize :: Storage -> CPtrdiff -> IO ()
  resize s pd = withForeignPtr (storage s) $ \s' -> Sig.c_resize s' pd

  fill :: Storage -> HsReal Storage -> IO ()
  fill s v = withForeignPtr (storage s) $ \s' -> Sig.c_fill s' (hs2cReal v)

