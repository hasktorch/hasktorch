{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Storage
  ( Storage(..)
  , asStorage
  ) where

import Foreign (Ptr, withForeignPtr, newForeignPtr)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import THTypes

import Torch.Core.Types
import Torch.Core.Storage.Copy ()
import qualified Storage as Sig
import qualified Torch.Class.Storage as Class
import qualified Foreign.Marshal.Array as FM

asStorage :: Ptr CStorage -> IO Storage
asStorage = fmap Storage . newForeignPtr Sig.p_free

instance Class.IsStorage Storage where
  tensordata :: Storage -> IO [HsReal]
  tensordata = ptrArray2hs Sig.c_data arrayLen . storage
   where
    arrayLen :: Ptr CStorage -> IO Int
    arrayLen p = case Sig.c_size p of { CPtrdiff pd -> pure (fromIntegral pd) }

  size :: Storage -> IO Int64
  size s = withForeignPtr (storage s) $ \s' -> do
    let CPtrdiff pd = Sig.c_size s'
    pure pd

  set :: Storage -> Int64 -> HsReal -> IO ()
  set s pd v = withForeignPtr (storage s) $ \s' -> Sig.c_set s' (CPtrdiff pd) (hs2cReal v)

  get :: Storage -> Int64 -> IO HsReal
  get s pd = withForeignPtr (storage s)  $ \s' -> pure . c2hsReal $ Sig.c_get s' (CPtrdiff pd)

  new :: IO Storage
  new = Sig.c_new >>= asStorage

  newWithSize :: Int64 -> IO Storage
  newWithSize pd = Sig.c_newWithSize (CPtrdiff pd) >>= asStorage

  newWithSize1 :: HsReal -> IO Storage
  newWithSize1 a0 = Sig.c_newWithSize1 (hs2cReal a0) >>= asStorage

  newWithSize2 :: HsReal -> HsReal -> IO Storage
  newWithSize2 a0 a1 = Sig.c_newWithSize2 (hs2cReal a0) (hs2cReal a1) >>= asStorage

  newWithSize3 :: HsReal -> HsReal -> HsReal -> IO Storage
  newWithSize3 a0 a1 a2 = Sig.c_newWithSize3 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) >>= asStorage

  newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> IO Storage
  newWithSize4 a0 a1 a2 a3 = Sig.c_newWithSize4 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3) >>= asStorage

  newWithMapping :: [Int8] -> Int64 -> Int32 -> IO Storage
  newWithMapping pcc' pd ci = do
    pcc <- FM.newArray (map CChar pcc')
    s <- Sig.c_newWithMapping pcc (CPtrdiff pd) (CInt ci)
    asStorage s

  newWithData :: [HsReal] -> Int64 -> IO Storage
  newWithData pr pd = do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    s <- Sig.c_newWithData pr' (CPtrdiff pd)
    asStorage s

  newWithAllocator :: Int64 -> CTHAllocatorPtr -> Ptr () -> IO Storage
  newWithAllocator pd calloc whatthehell = Sig.c_newWithAllocator (CPtrdiff pd) calloc whatthehell >>= asStorage

  newWithDataAndAllocator :: [HsReal] -> Int64 -> CTHAllocatorPtr -> Ptr () -> IO Storage
  newWithDataAndAllocator pr pd thalloc whatthehell = do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    s <- Sig.c_newWithDataAndAllocator pr' (CPtrdiff pd) thalloc whatthehell
    asStorage s

  setFlag :: Storage -> Int8 -> IO ()
  setFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_setFlag s' (CChar cc)

  clearFlag :: Storage -> Int8 -> IO ()
  clearFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_clearFlag s' (CChar cc)

  retain :: Storage -> IO ()
  retain s = withForeignPtr (storage s) Sig.c_retain

  swap :: Storage -> Storage -> IO ()
  swap s0 s1 =
    withForeignPtr (storage s0) $ \s0' ->
      withForeignPtr (storage s1) $ \s1' ->
        Sig.c_swap s0' s1'

  resize :: Storage -> Int64 -> IO ()
  resize s pd = withForeignPtr (storage s) $ \s' -> Sig.c_resize s' (CPtrdiff pd)

  fill :: Storage -> HsReal -> IO ()
  fill s v = withForeignPtr (storage s) $ \s' -> Sig.c_fill s' (hs2cReal v)

