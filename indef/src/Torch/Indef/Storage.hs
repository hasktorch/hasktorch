-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Storage
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Storages are basically a way to access memory of a C pointer or array.
-- Storages can also map the contents of a file to memory. A Storage is an
-- array of basic C types.
--
-- Several Storage classes for all the basic C types exist and have the
-- following self-explanatory names: ByteStorage, CharStorage, ShortStorage,
-- IntStorage, LongStorage, FloatStorage, DoubleStorage.
--
-- Note that ByteStorage and CharStorage represent both arrays of bytes.
-- ByteStorage represents an array of unsigned chars, while CharStorage
-- represents an array of signed chars.
-------------------------------------------------------------------------------
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

-- | return the internal data of 'Storage' as a list of haskell values.
tensordata :: Storage -> IO [HsReal]
tensordata s = withStorageState s $ \st s' ->
  ptrArray2hs (Sig.c_data st) (arrayLen st) (Sig.cstorage s)
 where
  arrayLen :: Ptr CState -> Ptr CStorage -> IO Int
  arrayLen st p = fromIntegral <$> Sig.c_size st p

-- | Returns the number of elements in the storage. Equivalent to #.
size :: Storage -> IO StorageSize
size s = withStorageState s $ \st s' -> fromIntegral <$> (Sig.c_size st s')

-- | set the value at 'Index' to 'HsReal' in a given 'Storage'.
set :: Storage -> Index -> HsReal -> IO ()
set s pd v = withStorageState s  $ \st s' -> Sig.c_set st s' (fromIntegral pd) (hs2cReal v)

-- | get the value at 'Index' from a given 'Storage'.
get :: Storage -> Index -> IO HsReal
get s pd = withStorageState s  $ \st s' -> c2hsReal <$> Sig.c_get st s' (fromIntegral pd)

-- | make a new empty 'Storage'.
empty :: IO Storage
empty = mkStorageIO Sig.c_new

-- | create a new storage of a given length, 'StorageSize'.
newWithSize :: StorageSize -> IO Storage
newWithSize pd = mkStorageIO $ \st -> Sig.c_newWithSize st (fromIntegral pd)

-- | make a new 'Storage' with a single value.
newWithSize1 :: HsReal -> IO Storage
newWithSize1 a0 = mkStorageIO $ \st -> Sig.c_newWithSize1 st (hs2cReal a0)

-- | make a new 'Storage' with two values.
newWithSize2 :: HsReal -> HsReal -> IO Storage
newWithSize2 a0 a1 = mkStorageIO $ \st -> Sig.c_newWithSize2 st (hs2cReal a0) (hs2cReal a1)

-- | make a new 'Storage' with three values.
newWithSize3 :: HsReal -> HsReal -> HsReal -> IO Storage
newWithSize3 a0 a1 a2 = mkStorageIO $ \st ->
  Sig.c_newWithSize3 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2)

-- | make a new 'Storage' with four values.
newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> IO Storage
newWithSize4 a0 a1 a2 a3 = mkStorageIO $ \st ->
  Sig.c_newWithSize4 st (hs2cReal a0) (hs2cReal a1) (hs2cReal a2) (hs2cReal a3)

-- | FIXME: This is totally broken. This takes a filename, size, and flags, and produces
-- 'Storage' from these inputs. Figure out how to fix this, ideally.
--
-- See:
-- https://github.com/torch/torch7/blob/04e1d1dce0f02aea82dc433c4f39e42650c4390f/lib/TH/generic/THStorage.h#L49
newWithMapping
  :: [Int8]       -- ^ filename
  -> StorageSize  -- ^ size
  -> Int32        -- ^ flags
  -> IO Storage
newWithMapping pcc' pd ci = mkStorageIO $ \st -> do
  pcc <- FM.newArray (map fromIntegral pcc')
  Sig.c_newWithMapping st pcc (fromIntegral pd) (fromIntegral ci)

-- | make a new 'Storage' from a given list and 'StorageSize'.
--
-- FIXME: find out if 'StorageSize' always corresponds to the length of the list. If so,
-- remove it!
newWithData :: [HsReal] -> StorageSize -> IO Storage
newWithData pr pd = mkStorageIO $ \st -> do
  pr' <- FM.withArray (hs2cReal <$> pr) pure
  Sig.c_newWithData st pr' (fromIntegral pd)

-- | Convenience method for 'newWithData'
fromList :: [HsReal] -> IO Storage
fromList pr = newWithData pr (fromIntegral $ length pr)

-- | set the flags of a given 'Storage'. Flags are applied via bitwise-or.
setFlag :: Storage -> Int8 -> IO ()
setFlag s cc = withStorageState s $ \st s' -> Sig.c_setFlag st s' (fromIntegral cc)

-- | clear the flags of a given 'Storage'. Flags are cleanred via bitwise-and.
clearFlag :: Storage -> Int8 -> IO ()
clearFlag s cc = withStorageState s $ \st s' -> Sig.c_clearFlag st s' (fromIntegral cc)

-- | Increment the reference counter of the storage.
--
-- This method should be used with extreme care. In general, they should never be called,
-- except if you know what you are doing, as the handling of references is done
-- automatically. They can be useful in threaded environments. Note that these
-- methods are atomic operations.
retain :: Storage -> IO ()
retain s = withStorageState s Sig.c_retain

-- | Resize the storage to the provided size. /The new contents are undetermined/.
resize :: Storage -> StorageSize -> IO ()
resize s pd = withStorageState s $ \st s' -> Sig.c_resize st s' (fromIntegral pd)

-- | Fill the Storage with the given value.
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
