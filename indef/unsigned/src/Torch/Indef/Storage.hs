{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Storage
  ( Storage
  , storage
  , asStorageM
  ) where

import Foreign (Ptr, withForeignPtr, newForeignPtr)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Torch.Types.TH
import Control.Monad.Managed

import Torch.Indef.Types
import Torch.Indef.Storage.Copy ()
import Torch.Class.Storage (Index(..), StorageSize(..), AllocatorContext(..))
import qualified Torch.Sig.Storage as Sig
import qualified Torch.Class.Storage as Class
import qualified Foreign.Marshal.Array as FM

asStorageM :: Ptr CStorage -> IO Storage
asStorageM = fmap asStorage . newForeignPtr Sig.p_free

instance Class.IsStorage Storage where
  tensordata :: Storage -> IO [HsReal]
  tensordata = ptrArray2hs Sig.c_data arrayLen . storage
   where
    arrayLen :: Ptr CStorage -> IO Int
    arrayLen p = fromIntegral <$> Sig.c_size p

  size :: Storage -> IO StorageSize
  size s = withForeignPtr (storage s) (fmap fromIntegral . Sig.c_size)

  set :: Storage -> Index -> HsReal -> IO ()
  set s pd v = withForeignPtr (storage s) $ \s' -> Sig.c_set s' (fromIntegral pd) (hs2cReal v)

  get :: Storage -> Index -> IO HsReal
  get s pd = withForeignPtr (storage s)  $ \s' -> c2hsReal <$> Sig.c_get s' (fromIntegral pd)

  empty :: IO Storage
  empty = Sig.c_new >>= asStorageM

  newWithSize :: StorageSize -> IO Storage
  newWithSize pd = Sig.c_newWithSize (fromIntegral pd) >>= asStorageM

  newWithSize1 :: HsReal -> IO Storage
  newWithSize1 a0 = Sig.c_newWithSize1 (hs2cReal a0) >>= asStorageM

  newWithSize2 :: HsReal -> HsReal -> IO Storage
  newWithSize2 a0 a1 = Sig.c_newWithSize2 (hs2cReal a0) (hs2cReal a1) >>= asStorageM

  newWithSize3 :: HsReal -> HsReal -> HsReal -> IO Storage
  newWithSize3 a0 a1 a2 = Sig.c_newWithSize3 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) >>= asStorageM

  newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> IO Storage
  newWithSize4 a0 a1 a2 a3 = Sig.c_newWithSize4 (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3) >>= asStorageM

  newWithMapping :: [Int8] -> StorageSize -> Int32 -> IO Storage
  newWithMapping pcc' pd ci = do
    pcc <- FM.newArray (map fromIntegral pcc')
    s <- Sig.c_newWithMapping pcc (fromIntegral pd) (fromIntegral ci)
    asStorageM s

  newWithData :: [HsReal] -> StorageSize -> IO Storage
  newWithData pr pd = do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    s <- Sig.c_newWithData pr' (fromIntegral pd)
    asStorageM s

  newWithAllocator :: StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
  newWithAllocator pd (alloc, AllocatorContext ctx) = Sig.c_newWithAllocator (fromIntegral pd) alloc ctx >>= asStorageM

  newWithDataAndAllocator :: [HsReal] -> StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
  newWithDataAndAllocator pr pd (alloc, AllocatorContext ctx) = do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    s <- Sig.c_newWithDataAndAllocator pr' (fromIntegral pd) alloc ctx {-seems like it's fine to pass nullPtr-}
    asStorageM s

  setFlag :: Storage -> Int8 -> IO ()
  setFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_setFlag s' (fromIntegral cc)

  clearFlag :: Storage -> Int8 -> IO ()
  clearFlag s cc = withForeignPtr (storage s) $ \s' -> Sig.c_clearFlag s' (fromIntegral cc)

  retain :: Storage -> IO ()
  retain s = withForeignPtr (storage s) Sig.c_retain

  swap :: Storage -> Storage -> IO ()
  swap s0 s1 =
    withForeignPtr (storage s0) $ \s0' ->
      withForeignPtr (storage s1) $ \s1' ->
        Sig.c_swap s0' s1'

  resize :: Storage -> StorageSize -> IO ()
  resize s pd = withForeignPtr (storage s) $ \s' -> Sig.c_resize s' (fromIntegral pd)

  fill :: Storage -> HsReal -> IO ()
  fill s v = withForeignPtr (storage s) $ \s' -> Sig.c_fill s' (hs2cReal v)

