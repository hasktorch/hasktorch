{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Storage
  ( module X
  , tensordata
  , size
  , set
  , get
  , empty
  , newWithSize
  , newWithSize1
  , newWithSize2
  , newWithSize3
  , newWithSize4
  , newWithMapping
  , newWithData
  , setFlag
  , clearFlag
  , retain
  , resize
  , fill
  ) where

import Torch.Indef.Storage.Copy as X

import Foreign
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Control.Monad
import Control.Monad.Managed

import Torch.Indef.Types
import Torch.Indef.Internal

import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.Types.Global   as Sig
import qualified Torch.Sig.Storage        as Sig
import qualified Torch.Sig.Storage.Memory as Sig
import qualified Foreign.Marshal.Array    as FM


tensordata :: Storage -> IO [HsReal]
tensordata s = withStorageState s $ \st s' ->
  ptrArray2hs (Sig.c_data st) (arrayLen st) (Sig.cstorage s)
 where
  arrayLen :: Ptr CState -> Ptr CStorage -> IO Int
  arrayLen st p = fromIntegral <$> Sig.c_size st p

size :: Storage -> IO StorageSize
size s = withStorageState s $ \st s' -> fromIntegral <$> (Sig.c_size st s')

set :: Storage -> Index -> HsReal -> IO ()
set s pd v = withStorageState s  $ \st s' -> Sig.c_set st s' (fromIntegral pd) (hs2cReal v)

get :: Storage -> Index -> IO HsReal
get s pd = withStorageState s  $ \st s' -> c2hsReal <$> Sig.c_get st s' (fromIntegral pd)

empty :: IO Storage
empty = mkStorageIO Sig.c_new

newWithSize :: StorageSize -> IO Storage
newWithSize pd = mkStorageIO $ \st -> Sig.c_newWithSize st (fromIntegral pd)

newWithSize1 :: HsReal -> IO Storage
newWithSize1 a0 = mkStorageIO $ \st -> Sig.c_newWithSize1 st (hs2cReal a0)

newWithSize2 :: HsReal -> HsReal -> IO Storage
newWithSize2 a0 a1 = mkStorageIO $ \st -> Sig.c_newWithSize2 st (hs2cReal a0) (hs2cReal a1)

newWithSize3 :: HsReal -> HsReal -> HsReal -> IO Storage
newWithSize3 a0 a1 a2 = mkStorageIO $ \st ->
  Sig.c_newWithSize3 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2)

newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> IO Storage
newWithSize4 a0 a1 a2 a3 = mkStorageIO $ \st ->
  Sig.c_newWithSize4 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3)

newWithMapping :: [Int8] -> StorageSize -> Int32 -> IO Storage
newWithMapping pcc' pd ci = mkStorageIO $ \st -> do
  pcc <- FM.newArray (map fromIntegral pcc')
  Sig.c_newWithMapping st pcc (fromIntegral pd) (fromIntegral ci)

newWithData :: [HsReal] -> StorageSize -> IO Storage
newWithData pr pd = mkStorageIO $ \st -> do
  pr' <- FM.withArray (hs2cReal <$> pr) pure
  Sig.c_newWithData st pr' (fromIntegral pd)

setFlag :: Storage -> Int8 -> IO ()
setFlag s cc = withStorageState s $ \st s' -> Sig.c_setFlag st s' (fromIntegral cc)

clearFlag :: Storage -> Int8 -> IO ()
clearFlag s cc = withStorageState s $ \st s' -> Sig.c_clearFlag st s' (fromIntegral cc)

retain :: Storage -> IO ()
retain s = withStorageState s Sig.c_retain

resize :: Storage -> StorageSize -> IO ()
resize s pd = withStorageState s $ \st s' -> Sig.c_resize st s' (fromIntegral pd)

fill :: Storage -> Sig.HsReal -> IO ()
fill s v = withStorageState s $ \st s' -> Sig.c_fill st s' (hs2cReal v)

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
