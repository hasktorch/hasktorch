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
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Storage
  ( IsList(..)
  , Storage(..)

  , cstorage
  , storage
  , storageState
  , storageStateRef

  , storagedata
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

import Foreign hiding (with, new)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import GHC.Word
import Control.Monad
import Control.Monad.Managed
import Control.DeepSeq
import System.IO.Unsafe
import GHC.Exts (IsList(..))
import Control.Monad.ST
import Data.STRef

import Torch.Indef.Types
import Torch.Indef.Internal

import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.Types.Global   as Sig
import qualified Torch.Sig.Storage        as Sig
import qualified Torch.Sig.Storage.Memory as Sig
import qualified Foreign.Marshal.Array    as FM

-- TODO: use these?
-- newtype MStorage s = MStorage (STRef s Storage)
--
-- freeze :: (forall s . STRef s Storage) -> Storage
-- freeze ref = runST (readSTRef ref)
--
-- thaw :: Storage -> ST s (MStorage s)
-- thaw s = MStorage <$> newSTRef s


-- | return the internal data of 'Storage' as a list of haskell values.
storagedata :: Storage -> [HsReal]
storagedata s = unsafeDupablePerformIO . flip with (pure . fmap c2hsReal) $ do
  st <- managedState
  s' <- managedStorage s
  liftIO $ do
    -- a strong dose of paranoia
    sz <- fromIntegral <$> Sig.c_size st s'
    tmp <- FM.mallocArray sz

    creals <- Sig.c_data st s'
    FM.copyArray tmp creals sz
    FM.peekArray sz tmp
 where
  arrayLen :: Ptr CState -> Ptr CStorage -> IO Int
  arrayLen st p = fromIntegral <$> Sig.c_size st p
{-# NOINLINE storagedata #-}

-- | Returns the number of elements in the storage. Equivalent to #.
size :: Storage -> Int
size s = unsafeDupablePerformIO . fmap fromIntegral . withLift $ Sig.c_size
  <$> managedState
  <*> managedStorage s
{-# NOINLINE size #-}

-- | set the value at 'Index' to 'HsReal' in a given 'Storage'.
set :: Storage -> Word -> HsReal -> IO ()
set s pd v = withLift $ Sig.c_set
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> pure (hs2cReal v)

-- | get the value at 'Index' from a given 'Storage'.
get :: Storage -> Word -> HsReal
get s pd = unsafeDupablePerformIO . fmap c2hsReal . withLift $ Sig.c_get
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
{-# NOINLINE get #-}

-- | make a new empty 'Storage'.
empty :: Storage
empty = unsafeDupablePerformIO . withStorage $ Sig.c_new
  <$> managedState
{-# NOINLINE empty #-}

-- | create a new storage of a given length, 'StorageSize'.
newWithSize :: Word -> Storage
newWithSize pd = unsafeDupablePerformIO . withStorage $ Sig.c_newWithSize
  <$> managedState
  <*> pure (fromIntegral pd)
{-# NOINLINE newWithSize #-}

-- | make a new 'Storage' with a single value.
newWithSize1 :: HsReal -> Storage
newWithSize1 a0 = unsafeDupablePerformIO . withStorage $ Sig.c_newWithSize1
  <$> managedState
  <*> pure (hs2cReal a0)
{-# NOINLINE newWithSize1 #-}

-- | make a new 'Storage' with two values.
newWithSize2 :: HsReal -> HsReal -> Storage
newWithSize2 a0 a1 = unsafeDupablePerformIO . withStorage $ Sig.c_newWithSize2
  <$> managedState
  <*> pure (hs2cReal a0)
  <*> pure (hs2cReal a1)
{-# NOINLINE newWithSize2 #-}

-- | make a new 'Storage' with three values.
newWithSize3 :: HsReal -> HsReal -> HsReal -> Storage
newWithSize3 a0 a1 a2 = unsafeDupablePerformIO . withStorage $ Sig.c_newWithSize3
  <$> managedState
  <*> pure (hs2cReal a0)
  <*> pure (hs2cReal a1)
  <*> pure (hs2cReal a2)
{-# NOINLINE newWithSize3 #-}

-- | make a new 'Storage' with four values.
newWithSize4 :: HsReal -> HsReal -> HsReal -> HsReal -> Storage
newWithSize4 a0 a1 a2 a3 = unsafeDupablePerformIO . withStorage $ Sig.c_newWithSize4
  <$> managedState
  <*> pure (hs2cReal a0)
  <*> pure (hs2cReal a1)
  <*> pure (hs2cReal a2)
  <*> pure (hs2cReal a3)
{-# NOINLINE newWithSize4 #-}

-- | FIXME: This is totally broken. This takes a filename, size, and flags, and produces
-- 'Storage' from these inputs. Figure out how to fix this, ideally.
--
-- See:
-- https://github.com/torch/torch7/blob/04e1d1dce0f02aea82dc433c4f39e42650c4390f/lib/TH/generic/THStorage.h#L49
newWithMapping
  :: [Int8]       -- ^ filename
  -> Word64       -- ^ size
  -> Int32        -- ^ flags
  -> IO Storage
newWithMapping pcc' pd ci = withStorage $ Sig.c_newWithMapping
  <$> managedState
  <*> liftIO (FM.newArray (map fromIntegral pcc'))
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral ci)

-- | make a new 'Storage' from a given list and 'StorageSize'.
--
-- FIXME: find out if 'StorageSize' always corresponds to the length of the list. If so,
-- remove it!
newWithData
  :: [HsReal]
  -> Word64   -- ^ storage size
  -> Storage
newWithData pr pd = unsafeDupablePerformIO . withStorage $ Sig.c_newWithData
  <$> managedState
  <*> liftIO (FM.newArray (hs2cReal <$> pr))
  <*> pure (fromIntegral pd)
{-# NOINLINE newWithData #-}

-- | set the flags of a given 'Storage'. Flags are applied via bitwise-or.
setFlag :: Storage -> Int8 -> IO ()
setFlag s cc = withLift $ Sig.c_setFlag
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral cc)

-- | clear the flags of a given 'Storage'. Flags are cleanred via bitwise-and.
clearFlag :: Storage -> Int8 -> IO ()
clearFlag s cc = withLift $ Sig.c_clearFlag
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral cc)

-- | Increment the reference counter of the storage.
--
-- This method should be used with extreme care. In general, they should never be called,
-- except if you know what you are doing, as the handling of references is done
-- automatically. They can be useful in threaded environments. Note that these
-- methods are atomic operations.
retain :: Storage -> IO ()
retain s = withLift $ Sig.c_retain
  <$> managedState
  <*> managedStorage s

-- | Resize the storage to the provided size. /The new contents are undetermined/.
resize :: Storage -> Word32 -> IO ()
resize s pd = withLift $ Sig.c_resize
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)

-- | Fill the Storage with the given value.
fill :: Storage -> HsReal -> IO ()
fill s v = withLift $ Sig.c_fill
  <$> managedState
  <*> managedStorage s
  <*> pure (hs2cReal v)

instance IsList Storage where
  type Item Storage = HsReal
  toList = storagedata
  fromList pr = newWithData pr (fromIntegral $ length pr)

instance Show Storage where
  show = show . storagedata

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
