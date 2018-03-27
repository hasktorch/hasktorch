{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Storage
  ( Storage
  , storage
  , mkStorage
  ) where

import Foreign
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Control.Monad
import Control.Monad.Managed
import Control.Monad.Reader.Class
import Control.Monad.IO.Class

import Torch.Indef.Internal
import Torch.Indef.Types
import Torch.Class.Storage (Index(..), StorageSize(..), AllocatorContext(..))

import qualified Torch.Class.Storage      as Class
import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.State          as Sig
import qualified Torch.Sig.Storage        as Sig
import qualified Torch.Sig.Storage.Memory as Sig
import qualified Foreign.Marshal.Array    as FM

mkStorage :: Ptr CStorage -> Torch Storage
mkStorage cstore = withState $ \st ->
  asStorage <$> newForeignPtrEnv Sig.p_free st cstore

mkStorageIO :: (Ptr CState -> IO (Ptr CStorage)) -> Torch Storage
mkStorageIO fn = withState fn >>= mkStorage

withState :: (Ptr CState -> IO x) -> Torch x
withState fn = ask >>= \st ->
  liftIO $ withForeignPtr (Sig.asForeign st) fn

withBoth :: Storage -> (Ptr CState -> Ptr CStorage -> IO x) -> Torch x
withBoth s fn = withState $ \stateP -> withForeignPtr (Sig.storage s) (fn stateP)

instance Class.Storage Torch Storage where
  tensordata :: Storage -> Torch [HsReal]
  tensordata s = withBoth s $ \st s' ->
    ptrArray2hs (Sig.c_data st) (arrayLen st) (Sig.storage s)
   where
    arrayLen :: Ptr CState -> Ptr CStorage -> IO Int
    arrayLen st p = fromIntegral <$> Sig.c_size st p

  size :: Storage -> Torch StorageSize
  size s = withBoth s $ \st s' -> fromIntegral <$> (Sig.c_size st s')

  set :: Storage -> Index -> HsReal -> Torch ()
  set s pd v = withBoth s  $ \st s' -> Sig.c_set st s' (fromIntegral pd) (hs2cReal v)

  get :: Storage -> Index -> Torch HsReal
  get s pd = withBoth s  $ \st s' -> c2hsReal <$> Sig.c_get st s' (fromIntegral pd)

  empty :: Torch Storage
  empty = mkStorageIO $ \st -> Sig.c_new st

  newWithSize :: StorageSize -> Torch Storage
  newWithSize pd = mkStorageIO $ \st -> Sig.c_newWithSize st (fromIntegral pd)

  newWithSize1 :: HsReal -> Torch Storage
  newWithSize1 a0 = mkStorageIO $ \st -> Sig.c_newWithSize1 st (hs2cReal a0)

  newWithSize2 :: HsReal -> HsReal -> Torch Storage
  newWithSize2 a0 a1 = mkStorageIO $ \st -> Sig.c_newWithSize2 st (hs2cReal a0) (hs2cReal a1)

  newWithSize3 :: HsReal -> HsReal -> HsReal -> Torch Storage
  newWithSize3 a0 a1 a2 = mkStorageIO $ \st ->
    Sig.c_newWithSize3 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2)

  newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> Torch Storage
  newWithSize4 a0 a1 a2 a3 = mkStorageIO $ \st ->
    Sig.c_newWithSize4 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3)

  newWithMapping :: [Int8] -> StorageSize -> Int32 -> Torch Storage
  newWithMapping pcc' pd ci = mkStorageIO $ \st -> do
    pcc <- FM.newArray (map fromIntegral pcc')
    Sig.c_newWithMapping st pcc (fromIntegral pd) (fromIntegral ci)

  newWithData :: [HsReal] -> StorageSize -> Torch Storage
  newWithData pr pd = mkStorageIO $ \st -> do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    Sig.c_newWithData st pr' (fromIntegral pd)

  setFlag :: Storage -> Int8 -> Torch ()
  setFlag s cc = withBoth s $ \st s' -> Sig.c_setFlag st s' (fromIntegral cc)

  clearFlag :: Storage -> Int8 -> Torch ()
  clearFlag s cc = withBoth s $ \st s' -> Sig.c_clearFlag st s' (fromIntegral cc)

  retain :: Storage -> Torch ()
  retain s = withBoth s Sig.c_retain

  resize :: Storage -> StorageSize -> Torch ()
  resize s pd = withBoth s $ \st s' -> Sig.c_resize st s' (fromIntegral pd)

  fill :: Storage -> Sig.HsReal -> Torch ()
  fill s v = withBoth s $ \st s' -> Sig.c_fill st s' (hs2cReal v)

{-
-- FIXME: find out where signatures should go to fill in these indefinites

instance Class.CPUStorage Storage where
  newWithAllocator :: StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
  newWithAllocator pd (alloc, AllocatorContext ctx) = Sig.c_newWithAllocator (fromIntegral pd) alloc ctx >>= mkStorage

  newWithDataAndAllocator :: [HsReal] -> StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
  newWithDataAndAllocator pr pd (alloc, AllocatorContext ctx) = do
    pr' <- FM.withArray (hs2cReal <$> pr) pure
    s <- Sig.c_newWithDataAndAllocator pr' (fromIntegral pd) alloc ctx {-seems like it's fine to pass nullPtr-}
    mkStorage s

  swap :: Storage -> Storage -> IO ()
  swap s0 s1 =
    withForeignPtr (storage s0) $ \s0' ->
      withForeignPtr (storage s1) $ \s1' ->
        Sig.c_swap s0' s1'


instance Class.GPUStorage t where
  c_getDevice :: t -> io Int
-}
