-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- This package is the class for handling numeric data in dynamic tensors.
--
-- A 'Dynamic' is a multi-dimensional matrix without static type-level
-- dimensions. The number of dimensions is unlimited (up to what can be created
-- using LongStorage).
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor where

import Foreign hiding (with, new)
import Foreign.Ptr
import Control.Applicative ((<|>))
import Control.Monad
import Control.Monad.Trans.Class
import Control.Monad.Managed
import Control.Exception.Safe
import Control.DeepSeq
import Data.Coerce (coerce)
import Data.Typeable
import Data.Maybe (fromMaybe)
import Data.List (intercalate, genericLength)
import Data.List.NonEmpty (NonEmpty(..), toList)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Numeric.Dimensions
import System.IO.Unsafe
import Control.Concurrent
import Control.Monad.Trans.Except

import Text.Printf
import qualified Data.List as List ((!!))
import qualified Data.List.NonEmpty        as NE
import qualified Torch.Types.TH            as TH
import qualified Foreign.Marshal.Array     as FM
import qualified Torch.Sig.State           as Sig
import qualified Torch.Sig.Types           as Sig
import qualified Torch.Sig.Types.Global    as Sig
import qualified Torch.Sig.Tensor          as Sig
import qualified Torch.Sig.Tensor.Memory   as Sig
import qualified Torch.Sig.Storage         as StorageSig (c_size)

import Torch.Indef.Types
import Torch.Indef.Internal
import Torch.Indef.Index hiding (withDynamicState)
import qualified Torch.Indef.Storage as Storage

instance Show Dynamic where
  show t = unsafePerformIO $ do
    SomeDims ds <- getDims t
    (vs, desc) <- showTensor
      (get1d t) (get2d t) (\a b c -> get3d t a b c) (\a b c d -> get4d t a b c d)
      (fromIntegral <$> listDims ds)
    pure (vs ++ "\n" ++ desc)
  {-# NOINLINE show #-}


-- CPU ONLY:
--   desc :: Dynamic -> IO (DescBuff t)

-- | Clears the internal flags on a tensor. Uses bitwise operators for flags.
_clearFlag :: Dynamic -> Int8 -> IO ()
_clearFlag t cc = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_clearFlag s' t' (CChar cc)

-- | get the underlying data as a haskell list from the tensor
--
-- NOTE: This _cannot_ use a Tensor's storage size because ATen's Storage
-- allocates up to the next 64-byte line on the CPU (needs reference, this
-- is the unofficial response from \@soumith in slack).
#ifndef HASKTORCH_INTERNAL_CUDA
tensordata :: Dynamic -> IO [HsReal]
tensordata t = flip with pure $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ do
    ds <- shape t
    let sz = fromIntegral $ product ds
    creals <- Sig.c_data s' t'
    (fmap.fmap) c2hsReal (FM.peekArray sz creals)
#endif

-- | get a value from dimension 1
get1d :: Dynamic -> Int64 -> IO HsReal
get1d t d1 = flip with (pure . c2hsReal) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_get1d s t' (fromIntegral d1)
{-# NOINLINE get1d #-}

-- | get a value from dimension 2
get2d :: Dynamic -> Int64 -> Int64 -> IO HsReal
get2d t d1 d2 = flip with (pure . c2hsReal) $ do
  s  <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_get2d s t' (fromIntegral d1) (fromIntegral d2)
{-# NOINLINE get2d #-}

-- | get a value from dimension 3
get3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO HsReal
get3d t d1 d2 d3 = flip with (pure . c2hsReal) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_get3d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)
{-# NOINLINE get3d #-}

-- | get a value from dimension 4
get4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
get4d t d1 d2 d3 d4 = flip with (pure . c2hsReal) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_get4d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3) (fromIntegral d4)
{-# NOINLINE get4d #-}

-- | whether or not the tensor is contiguous in memory.
isContiguous :: Dynamic -> IO Bool
isContiguous t = flip with (pure . (1 ==)) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_isContiguous s t'

-- | check to see if to tensors are the same size as eachother.
isSameSizeAs :: Dynamic -> Dynamic -> IO Bool
isSameSizeAs t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
  (1 ==) <$> Sig.c_isSetTo s t0' t1'

-- | Returns true iff the Tensor is set to the argument Tensor.
--
-- Note: this is only true if the tensors are the same size, have
-- the same strides and share the same storage and offset.
isSetTo :: Dynamic -> Dynamic -> IO Bool
isSetTo t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
  (1 ==) <$> Sig.c_isSetTo s t0' t1'

-- | check to see if the tensor is the same size as the LongStorage.
isSize :: Dynamic -> TH.LongStorage -> IO Bool
isSize t ls = flip with (pure . (1 ==)) $ do
  s <- managedState
  t' <- managedTensor t
  l' <- managed $ withForeignPtr (snd $ TH.longStorageState ls)
  liftIO $ Sig.c_isSize s t' l'

-- | Returns the number of dimensions in a Tensor.
nDimension :: Dynamic -> IO Word
nDimension t = flip with (pure . fromIntegral) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_nDimension s t'

-- | Returns the number of elements in a Tensor.
nElement :: Dynamic -> IO Int64
nElement t = flip with (pure . fromIntegral) $ do
  s <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_nElement s t'

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
_narrow :: Dynamic -> Dynamic -> DimVal -> Int64 -> Size -> IO ()
_narrow t0 t1 a b c = runManaged $ do
  s'  <- managedState
  t0' <- managedTensor t0
  t1' <- managedTensor t1
  liftIO $ Sig.c_narrow s' t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

-- | Returns an empty tensor.
empty :: IO Dynamic
empty = withForeignPtr Sig.torchstate $ Sig.c_new >=> mkDynamic
{-# NOINLINE empty #-}

-- | pure version of '_expand'
newExpand :: Dynamic -> TH.IndexStorage -> IO Dynamic
newExpand r ix = flip with pure $ do
  s <- managedState
  r' <- managedTensor r
  ix' <- managed (withForeignPtr . snd $ TH.longStorageState ix)
  liftIO $ Sig.c_newExpand s r' ix' >>= mkDynamic

-- | Expanding a tensor does not allocate new memory, but only creates a new view on the
-- existing tensor where singleton dimensions can be expanded to multiple ones by setting
-- the stride to 0. Any dimension that has size 1 can be expanded to arbitrary value
-- without any new memory allocation. Attempting to expand along a dimension that does
-- not have size 1 will result in an error which we do not currently handle in hasktorch.
_expand
  :: Dynamic          -- ^ return tensor to mutate inplace.
  -> Dynamic          -- ^ source tensor to expand
  -> TH.IndexStorage  -- ^ how to expand the tensor.
  -> IO ()
_expand r t ix = runManaged . (liftIO =<<) $ Sig.c_expand
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> managed (withForeignPtr . snd $ TH.longStorageState ix)

-- | FIXME: doublecheck what this does.
_expandNd  :: NonEmpty Dynamic -> NonEmpty Dynamic -> Int -> IO ()
_expandNd (rets@(s:|_)) ops i = runManaged $ do
  st    <- managedState
  rets' <- mngNonEmpty rets
  ops'  <- mngNonEmpty ops
  liftIO $ Sig.c_expandNd st rets' ops' (fromIntegral i)
 where
  mngNonEmpty :: NonEmpty Dynamic -> Managed (Ptr (Ptr CTensor))
  mngNonEmpty = mapM toMPtr . toList >=> mWithArray

  mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
  mWithArray as = managed (FM.withArray as)

  toMPtr :: Dynamic -> Managed (Ptr CTensor)
  toMPtr d = managed (withForeignPtr (Sig.ctensor d))

-- | purely clone a tensor
newClone :: Dynamic -> IO Dynamic
newClone t = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newClone s' t'

-- | purely clone a tensor to have a contiguous memory layout.
newContiguous :: Dynamic -> IO Dynamic
newContiguous t = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newContiguous s' t'

{-# WARNING newSelect, newNarrow, _set, _select, _narrow "hasktorch devs have not yet made this safe. You are warned." #-}
-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
newNarrow :: Dynamic -> DimVal -> Int64 -> Size -> IO Dynamic
newNarrow t a b c = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newNarrow s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c)

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
newSelect :: Dynamic -> DimVal -> Int64 -> IO Dynamic
newSelect t a b = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newSelect s' t' (fromIntegral a) (fromIntegral b)

-- | get the sizes of each dimension
--
-- FIXME: doublecheck this
newSizeOf :: Dynamic -> IO (TH.IndexStorage)
newSizeOf t = flip with mkCPUIxStorage $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newSizeOf s' t'

-- | get the strides of each dimension
--
-- FIXME: doublecheck this
newStrideOf :: Dynamic -> IO (TH.IndexStorage)
newStrideOf t = flip with mkCPUIxStorage $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newStrideOf s' t'

-- | pure version of '_transpose'
newTranspose :: Dynamic -> DimVal -> DimVal -> IO Dynamic
newTranspose t a b = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newTranspose s' t' (fromIntegral a) (fromIntegral b)

-- | pure version of '_unfold'
newUnfold :: Dynamic -> DimVal -> Int64 -> Int64 -> IO Dynamic
newUnfold t a b c = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newUnfold s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c)

-- |
-- Creates a view with different dimensions of the storage associated with tensor, returning a new tensor.
--
-- FIXME: I think resizeAs is the non-cloning version of this function. See:
-- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes
--
-- for more.
newView :: Dynamic -> TH.IndexStorage -> IO Dynamic
newView t ix = flip with mkDynamic $ do
  s'  <- managedState
  t'  <- managedTensor t
  ix' <- managed $ withCPUIxStorage ix
  liftIO $ Sig.c_newView s' t' ix'

-- | create an uninitialized tensor with the given size and strides (?)
--
-- FIXME: doublecheck what the IndexStorages stands for
newWithSize :: TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
newWithSize l0 l1 =
  withCPUIxStorage l0 $ \l0' ->
    withCPUIxStorage l1 $ \l1' ->
      mkDynamicIO $ \s ->
        Sig.c_newWithSize s l0' l1'

-- | create an uninitialized 1d tensor
newWithSize1d :: Size -> IO Dynamic
newWithSize1d a0 = mkDynamicIO $ \s -> Sig.c_newWithSize1d s (fromIntegral a0)

-- | create an uninitialized 2d tensor
newWithSize2d :: Size -> Size -> IO Dynamic
newWithSize2d a0 a1 = mkDynamicIO $ \s -> Sig.c_newWithSize2d s (fromIntegral a0) (fromIntegral a1)

-- | create an uninitialized 3d tensor
newWithSize3d :: Size -> Size -> Size -> IO Dynamic
newWithSize3d a0 a1 a2 = mkDynamicIO $ \s -> Sig.c_newWithSize3d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2)

-- | create an uninitialized 4d tensor
newWithSize4d :: Size -> Size -> Size -> Size -> IO Dynamic
newWithSize4d a0 a1 a2 a3 = mkDynamicIO $ \s -> Sig.c_newWithSize4d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)

{-# WARNING newWithStorage, newWithStorage1d, newWithStorage2d, newWithStorage3d, newWithStorage4d, newWithTensor "hasktorch devs have not yet made this safe. You are warned." #-}

{-# NOINLINE newWithStorage #-}
{-# NOINLINE newWithStorage1d #-}
{-# NOINLINE newWithStorage2d #-}
{-# NOINLINE newWithStorage3d #-}
{-# NOINLINE newWithStorage4d #-}
{-# NOINLINE newWithTensor  #-}
-- | create a new tensor with the given size and strides, storage offset and storage.
--
-- FIXME: doublecheck what all of this does.
newWithStorage :: Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
newWithStorage s pd l0 l1 =
  withStorageState s $ \state' s' ->
    withForeignPtr (snd $ TH.longStorageState l0) $ \l0' ->
      withForeignPtr (snd $ TH.longStorageState l1) $ \l1' ->
        Sig.c_newWithStorage state' s' (fromIntegral pd) l0' l1'
        >>= mkDynamic

-- | create a new 1d tensor with the given storage's first dimension.
newWithStorage1d
  :: Storage            -- storage to use
  -> StorageOffset      -- storage offset must be >= 1
  -> (Size, Stride)     -- size is of the 1st dimension, stride is the stride in the first dimension
  -> IO Dynamic
newWithStorage1d s pd (d00,d01) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage1d state' s' (fromIntegral pd)
    (fromIntegral d00) (fromIntegral d01)
    >>= mkDynamic


-- | create a new 2d tensor with the given storage's first 2 dimensions.
newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage2d s pd (d00,d01) (d10,d11) = flip with mkDynamic $ do
  state' <- managedState
  s'     <- managed (withForeignPtr (Sig.cstorage s))
  liftIO $ Sig.c_newWithStorage2d state' s'
    (fromIntegral pd)
    (fromIntegral d00) (fromIntegral d01)
    (fromIntegral d10) (fromIntegral d11)
  -- liftIO $ mkDynamic t'


-- | create a new 3d tensor with the given storage's first 3 dimensions.
newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage3d s pd (d00,d01) (d10,d11) (d20,d21) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage3d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
    >>= mkDynamic


-- | create a new 4d tensor with the given storage's first 4 dimensions.
newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage4d s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage4d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
      (fromIntegral d30) (fromIntegral d31)
    >>= mkDynamic

-- | create a new tensor with the given tensor's underlying storage.
newWithTensor :: Dynamic -> IO Dynamic
newWithTensor t = flip with mkDynamic $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newWithTensor s' t'

-- | Resize the tensor according to the given LongStorage size (and strides?)
-- FIXME: doublecheck what the IndexStorages stands for
_resize :: Dynamic -> TH.IndexStorage -> TH.IndexStorage -> IO ()
_resize t l0 l1 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  l0' <- managed $ withCPUIxStorage l0
  l1' <- managed $ withCPUIxStorage l1
  liftIO $ Sig.c_resize s' t' l0' l1'

-- | resize dimension 1 of a tensor.
_resize1d :: Dynamic -> Int64 -> IO ()
_resize1d t l0 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_resize1d s' t' (fromIntegral l0)

-- | resize the first 2 dimensions of a tensor.
_resize2d :: Dynamic -> Int64 -> Int64 -> IO ()
_resize2d t l0 l1 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_resize2d s' t'
    (fromIntegral l0) (fromIntegral l1)

-- | resize the first 3 dimensions of a tensor.
_resize3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO ()
_resize3d t l0 l1 l2 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_resize3d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2)

-- | resize the first 4 dimensions of a tensor.
_resize4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
_resize4d t l0 l1 l2 l3 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_resize4d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3)

-- | resize the first 5 dimensions of a tensor.
_resize5d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
_resize5d t l0 l1 l2 l3 l4 = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_resize5d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (fromIntegral l4)

-- | Resize the tensor as the given tensor.
_resizeAs :: Dynamic -> Dynamic -> IO ()
_resizeAs t0 t1 = with2DynamicState t0 t1 Sig.c_resizeAs

-- | resize a tensor with given strides, sizes and a magical parameter.
--
-- FIXME: Someone needs to find out what the magical parameter is.
_resizeNd
  :: Dynamic   -- ^ tensor to resize inplace.
  -> Int32     -- ^ unknown argument. FIXME: Someone needs to find this out.
  -> [Size]    -- ^ new sizes to update
  -> [Stride]  -- ^ new strides to update.
  -> IO ()
_resizeNd t i l0' l1' = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ do
    l0 <- FM.newArray (coerce l0' :: [CLLong])
    l1 <- FM.newArray (coerce l1' :: [CLLong])
    Sig.c_resizeNd s' t' (fromIntegral i) l0 l1

-- | Increment the reference counter of the tensor.
--
-- From: https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/tensor.md#reference-counting
--
-- Tensors are reference-counted. It means that each time an object (C or the Lua state) need to keep a reference over a tensor, the corresponding tensor reference counter will be increased. The reference counter is decreased when the object does not need the tensor anymore.
--
-- These methods should be used with extreme care. In general, they should never be called, except if you know what you are doing, as the handling of references is done automatically. They can be useful in threaded environments. Note that these methods are atomic operations.
retain :: Dynamic -> IO ()
retain t = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_retain s' t'

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
_select :: Dynamic -> Dynamic -> DimVal -> Int64 -> IO ()
_select t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_select s' t0' t1' (fromIntegral a) (fromIntegral b)

-- | set the source tensor's storage to another tensor.
_set
  :: Dynamic  -- ^ the source tensor which is mutated inplace
  -> Dynamic  -- ^ the tensor who's storage is going to be referenced.
  -> IO ()
_set t0 t1 = with2DynamicState t0 t1 Sig.c_set

-- | set a value in dimension 1, inplace.
_set1d :: Dynamic -> Int64 -> HsReal -> IO ()
_set1d t l0 v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_set1d s' t' (fromIntegral l0) (hs2cReal v)

-- | set a value in dimension 2, inplace.
_set2d :: Dynamic -> Int64 -> Int64 -> HsReal -> IO ()
_set2d t l0 l1 v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_set2d s' t' (fromIntegral l0) (fromIntegral l1) (hs2cReal v)

-- | set a value in dimension 3, inplace.
_set3d :: Dynamic -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
_set3d t l0 l1 l2 v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_set3d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (hs2cReal v)

-- | set a value in dimension 4, inplace.
_set4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
_set4d t l0 l1 l2 l3 v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_set4d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (hs2cReal v)

-- | set the flags on a tensor inplace
_setFlag :: Dynamic -> Int8 -> IO ()
_setFlag t l0 = flip with pure $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_setFlag s' t' (CChar l0)

-- | Set the storage of a tensor.
--
-- FIXME: doublecheck what the IndexStorages stands for
_setStorage :: Dynamic -> Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO ()
_setStorage t s a b c = withDynamicStateAndStorage t s $ \st' t' s' ->
  runManaged $ do
    b' <- managed $ withCPUIxStorage b
    c' <- managed $ withCPUIxStorage c
    liftIO $ Sig.c_setStorage st' t' s' (fromIntegral a) b' c'

-- | Set the storage of a tensor, only referencing 1 dimension of storage
--
-- FIXME: find out what storageoffset does.
_setStorage1d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
_setStorage1d t s pd (d00,d01) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage1d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)

-- | Set the storage of a tensor, only referencing 2 dimensions of storage
--
-- FIXME: find out what storageoffset does.
_setStorage2d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage2d t s pd (d00,d01) (d10,d11) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage2d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)


-- | Set the storage of a tensor, only referencing 3 dimensions of storage
--
-- FIXME: find out what storageoffset does.
_setStorage3d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage3d t s pd (d00,d01) (d10,d11) (d20,d21) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage3d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)

-- | Set the storage of a tensor, only referencing 4 dimensions of storage
--
-- FIXME: find out what storageoffset does.
_setStorage4d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage4d t s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage4d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
      (fromIntegral d30) (fromIntegral d31)

-- | Set the storage of a tensor, referencing any number of dimensions of storage
_setStorageNd :: Dynamic -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
_setStorageNd t s a b hsc hsd = do
  c <- FM.newArray (coerce hsc :: [CLLong])
  d <- FM.newArray (coerce hsd :: [CLLong])
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorageNd st' t' s' (fromIntegral a) (fromIntegral b) c d

-- | get the size of a tensor's specific dimension.
size :: Dynamic -> DimVal -> IO Word
size t l0 = flip with (pure . fromIntegral) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_size s' t' (fromIntegral l0)

-- | primarily used for debugging. Get the size description from a c call.
sizeDesc :: Dynamic -> IO DescBuff
sizeDesc t = flip with (Sig.descBuff) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_sizeDesc s' t'

-- | Removes all singleton dimensions of the tensor.
_squeeze :: Dynamic -> Dynamic -> IO ()
_squeeze t0 t1 = with2DynamicState t0 t1 Sig.c_squeeze

-- | Removes a singleton dimensions of the tensor at a given dimension.
_squeeze1d :: Dynamic -> Dynamic -> DimVal -> IO ()
_squeeze1d t0 t1 d = with2DynamicState t0 t1 (shuffle3 Sig.c_squeeze1d (fromIntegral d))

-- | get the underlying storage of a tensor
storage :: Dynamic -> IO Storage
storage t = flip with mkStorage $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_storage s' t'

-- | get the storage offset of a tensor
storageOffset :: Dynamic -> IO StorageOffset
storageOffset t = flip with (pure . fromIntegral) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_storageOffset s' t'

-- | Returns the jump necessary to go from one element to the next one in the
-- specified dimension dim.
stride :: Dynamic -> DimVal -> IO Stride
stride t a = flip with (pure . fromIntegral) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_stride s' t' (fromIntegral a)

-- | Returns a tensor where dimensions dim1 and dim2 have been swapped.
_transpose
  :: Dynamic  -- ^ tensor to mutate into the result.
  -> Dynamic  -- ^ source tensor to use for data.
  -> DimVal   -- ^ dim1
  -> DimVal   -- ^ dim2
  -> IO ()
_transpose t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_transpose s' t0' t1' (fromIntegral a) (fromIntegral b)

-- | Returns a tensor which contains all slices of size size in the dimension dim.
-- Step between two slices is given by step.
--
-- If sizedim is the original size of dimension dim, the size of dimension dim
-- in the returned tensor will be (sizedim - size) / step + 1
--
-- An additional dimension of size size is appended in the returned tensor.
--
-- FIXME: this still takes C-like arguments which mutates the first arg inplace.
_unfold
  :: Dynamic  -- ^ tensor to mutate into the result.
  -> Dynamic  -- ^ source tensor to use for data.
  -> DimVal -> Size -> Step -> IO ()
_unfold t0 t1 a b c = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_unfold s' t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

-- | unsqueeze a tensor, adding a singleton dimension at the specified dimval.
_unsqueeze1d
  :: Dynamic  -- ^ tensor to mutate into the result.
  -> Dynamic  -- ^ source tensor to use for data.
  -> DimVal   -- ^ dimension to unsqueeze
  -> IO ()
_unsqueeze1d t0 t1 d = with2DynamicState t0 t1 $
  shuffle3 Sig.c_unsqueeze1d (fromIntegral d)

-- ========================================================================= --
-- User API (can be bundled into the above)
-- ========================================================================= --

-- | return the a runtime shape representing the dimensions of a 'Dynamic'
shape :: Dynamic -> IO [Word]
shape t = nDimension t >>= \case
    0 -> pure []
    d -> mapM (size t . fromIntegral) [0.. d - 1]

-- | set the storage dimensionality of a dynamic tensor, inplace, to any new size and stride pair.
_setStorageDim :: Dynamic -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

-- | set a value of a dynamic tensor, inplace, with any dimensionality.
_setDim :: Dynamic -> Dims (d::[Nat]) -> HsReal -> IO ()
_setDim t d !v = do
 threadDelay 1000
 case fromIntegral <$> listDims d of
  []           -> throwNE "can't set on an empty dimension."
  [!x]          -> _set1d t x          v
  [!x, !y]       -> _set2d t x y       v
  [!x, !y, !z]    -> _set3d t x y z    v
  [!x, !y, !z, !q] -> _set4d t x y z q v
  _            -> throwGT4 "set"

-- | resize a dynamic tensor, inplace, to any new dimensionality
_resizeDim :: Dynamic -> Dims (d::[Nat]) -> IO ()
_resizeDim t d = case fromIntegral <$> listDims d of
  []              -> throwNE "can't resize to an empty dimension."
  [!x]             -> _resize1d t x
  [!x, !y]          -> _resize2d t x y
  [!x, !y, !z]       -> _resize3d t x y z
  [!x, !y, !z, !q]    -> _resize4d t x y z q
  [!x, !y, !z, !q, !w] -> _resize5d t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

-- | create a 1d Dynamic tensor from a list of elements.
--
-- FIXME construct this with TH, not by using 'setDim' inplace (one-by-one) which might be doing a second linear pass.
-- FIXME: CUDA doesn't like the storage allocation:

vector :: [HsReal] -> ExceptT String IO Dynamic
vector l = lift $ do
---------------------------------------------------
-- THCudaCheck FAIL file=/home/stites/git/hasktorch/vendor/aten/src/THC/generic/THCStorage.c line=150 error=11 : invalid argument
-- terminate called after throwing an instance of 'std::runtime_error'
--   what():  cuda runtime error (11) : invalid argument at /home/stites/git/hasktorch/vendor/aten/src/THC/generic/THCStorage.c:150
--   Aborted (core dumped)
-- #ifdef HASKTORCH_INTERNAL_CUDA

-- FIXME: uncomment these
  -- res <- new' (someDimsVal [genericLength l])
  -- mapM_  (upd res) (zip [0..genericLength l - 1] l)
  -- pure res

-- #else
--   -- IMPORTANT: This is safe for CPU. For GPU, I think we need to allocate and marshal a haskell list into cuda memory before continuing.
  st <- Storage.fromList (deepseq l l)
  newWithStorage1d st 0 (genericLength l, 1)
-- #endif
 where
  upd :: Dynamic -> (Word, HsReal) -> IO ()
  upd t (idx, !v) =
    let ix = [idx]
    in setDim'_ t (someDimsVal (deepseq ix ix)) v


-- | create a 2d Dynamic tensor from a list of list of elements.
matrix :: [[HsReal]] -> ExceptT String IO Dynamic
matrix ls
  | null ls = lift empty
  | any ((ncols /=) . length) ls = ExceptT . pure $ Left "rows are not all the same length"
  | otherwise = do
-- #ifdef HASKTORCH_INTERNAL_CUDA
    -- vec <- vector (deepseq l l)
    -- lift $ go vec (someDimsVal [nrows, ncols])
    -- pure vec
-- #else
    lift $ do
      st <- Storage.fromList (deepseq l l)
      newWithStorage2d st 0 (nrows, ncols) (ncols, 1)
-- #endif
 where
  l = concat ls
  go vec (SomeDims ds) = _resizeDim vec ds

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


-- | create a 3d Dynamic tensor (ie: rectangular cuboid) from a nested list of elements.
{-# NOINLINE cuboid #-}
cuboid :: [[[HsReal]]] -> ExceptT String IO Dynamic
cuboid ls
  | isEmpty ls = lift empty
  | null ls || any null ls || any (any null) ls
                                   = ExceptT . pure . Left $ "can't accept empty lists"
  | innerDimCheck ncols        ls  = ExceptT . pure . Left $ "rows are not all the same length"
  | innerDimCheck ndepth (head ls) = ExceptT . pure . Left $ "columns are not all the same length"

  | otherwise = lift $ do
      st <- Storage.fromList l
      -- FIXME: test that this is correct.
      newWithStorage3d st 0 (nrows, ncols * ndepth) (ncols, ndepth) (ndepth, 1)
-- #ifdef HASKTORCH_INTERNAL_CUDA
      -- vec <- vector (deepseq l l)
      -- lift $ go vec (someDimsVal [nrows, ncols, ndepth])
      -- lift $ go vec (someDimsVal [nrows, ncols, ndepth])
      -- print "yas"
      -- pure v
-- #else
--       -- st <- Storage.fromList (deepseq l l)
--       -- hs <- Storage.tensordata st
--
--       -- print (nrows, ncols, ndepth)
--       -- forM_ [0..nrows-1] $ \r -> do
--       -- -- forM_ [0..nrows-1] $ \r -> do
--       --   forM_ [0..ncols-1] $ \c -> do
--       --   -- forM_ [0..1] $ \c -> do
--       --     printf "\n]] "
--       --     forM_ [0..ndepth-1] $ \d -> do
--       --     -- forM_ [0..1] $ \d -> do
--       --       let v = hs List.!! ((r*ncols*ndepth) + (c*ndepth) + d)
--       --       -- let v = (x List.!! 0) List.!! r List.!! c :: HsReal
--       --       printf ((if v < 0 then " " else "  ")++"%.4f") (v :: HsReal)
--       --   putStrLn ("\n" :: String)
--       -- print "ax"
--
--       vec <- vector (deepseq l l)
--       lift $ go vec (someDimsVal [nrows, ncols, ndepth])
--
--
--       -- newWithStorage3d st 0 (nrows, ncols * ndepth) (ncols, ndepth) (ndepth, 1)
--       -- newWithStorage3d st 0 (nrows, nrows) (ncols, 1) (ndepth, 1)
--       -- newWithStorage3d st 0 (nrows, ncols*ndepth) (ncols, ndepth) (ndepth, 1)
-- #endif
 where
  l = concat (concat ls)
  go vec (SomeDims ds) = _resizeDim vec ds >> pure vec

  isEmpty = \case
    []     -> True
    [[]]   -> True
    [[[]]] -> True
    _      -> False

  innerDimCheck :: Int -> [[x]] -> Bool
  innerDimCheck d = any ((/= d) . length)

  ndepth :: Integral i => i
  ndepth = genericLength (head (head ls))

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


-- | create a 4d Dynamic tensor (ie: hyperrectangle) from a nested list of elements.
{-# NOINLINE hyper #-}
hyper :: [[[[HsReal]]]] -> ExceptT String IO Dynamic
hyper ls
  | isEmpty ls = lift empty
  | null ls
    || any null ls
    || any (any null) ls
    || any (any (any null)) ls           = ExceptT . pure . Left $ "can't accept empty lists"
  | innerDimCheck ntime (head (head ls)) = ExceptT . pure . Left $ "rows are not all the same length"
  | innerDimCheck ndepth      (head ls)  = ExceptT . pure . Left $ "cols are not all the same length"
  | innerDimCheck ncols             ls   = ExceptT . pure . Left $ "depths are not all the same length"

  | otherwise = lift $ do
-- #ifdef HASKTORCH_INTERNAL_CUDA
      -- vec <- vector (deepseq l l)
      -- lift $ go vec (someDimsVal [nrows, ncols, ndepth, ntime])
-- #else
      st <- Storage.fromList (deepseq l l)
      newWithStorage4d st 0
        (nrows, ncols * ndepth * ntime)
        (ncols, ndepth * ntime)
        (ndepth, ntime)
        (ntime, 1)
-- #endif
 where
  l = concat (concat (concat ls))
  go vec (SomeDims ds) = _resizeDim vec ds >> pure vec

  isEmpty = \case
    []       -> True
    [[]]     -> True
    [[[]]]   -> True
    [[[[]]]] -> True
    _        -> False

  innerDimCheck :: Int -> [[x]] -> Bool
  innerDimCheck d = any ((/= d) . length)

  ntime :: Integral i => i
  ntime = genericLength (head (head (head ls)))

  ndepth :: Integral i => i
  ndepth = genericLength (head (head ls))

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


-- | resize a dynamic tensor with runtime 'SomeDims' representation of its new shape. Returns a pure copy of the
-- input tensor.
resizeDim :: Dynamic -> Dims (d::[Nat]) -> IO Dynamic
resizeDim src d = newClone src >>= \res -> _resizeDim res d >> pure res

-- | alias to 'resizeDim' with a runtime 'SomeDims'
resizeDim' :: Dynamic -> SomeDims -> IO Dynamic
resizeDim' t (SomeDims d) = resizeDim t d

-- | get a value from a dynamic tensor at a given index, @Dims d@.
getDim :: Dynamic -> Dims (d::[Nat]) -> IO HsReal
getDim t d = case fromIntegral <$> listDims d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

-- | alias to 'getDimList' which wraps the dimensions list in a 'SomeDims'
getDims :: Dynamic -> IO SomeDims
getDims = fmap someDimsVal . getDimList

-- | get the runtime dimension list of a dynamic tensor
getDimList :: Integral i => Dynamic -> IO [i]
getDimList t = nDimension t >>= \n ->
  mapM (fmap fromIntegral . size t . fromIntegral) [0 .. n - 1]

-- | create a new dynamic tensor of size @Dims d@
new :: forall (d::[Nat]) . Dims d -> IO Dynamic
new d = case fromIntegral <$> listDims d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> do
    t <- empty
    _resizeDim t d
    pure t

-- | set a specific runtime 'SomeDims' position of a dynamic tensor.
setDim'_ :: Dynamic -> SomeDims -> HsReal -> IO ()
setDim'_ t (SomeDims d) v = _setDim t d v

-- | resize a dynamic tensor inplace with runtime 'SomeDims' representation of its new shape.
resizeDim'_ :: Dynamic -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = _resizeDim t d

-- | get a specific value of a dynamic tensor with runtime 'SomeDims' index
getDim' :: Dynamic -> SomeDims -> IO HsReal
getDim' t (SomeDims d) = getDim t d

-- | build a new tensor with a runtime 'SomeDims'
new' :: SomeDims -> IO Dynamic
new' (SomeDims d) = new d

-- | resize a tensor to take the shape of the second 'Dynamic' argument.
-- This is a pure function.
--
-- FIXME: Is this right? why are there three tensors
resizeAs
  :: Dynamic -- ^ src
  -> Dynamic -- ^ a tensor only used for its shape
  -> IO Dynamic  -- ^ a new copy of src with the shape tensor's shape
resizeAs src shape = do
  res <- newClone src
  _resizeAs res shape
  pure res

-- | Generic way of showing the internal data of a tensor in a tabular format.
-- This makes no assumptions about the type of representation to show and can be
-- used for 'Storage', 'Dynamic', and 'Tensor' types.
showTensor
  :: forall a . (Typeable a, Ord a, Num a, Show a)
  => (Int64 -> IO a)
  -> (Int64 -> Int64 -> IO a)
  -> (Int64 -> Int64 -> Int64 -> IO a)
  -> (Int64 -> Int64 -> Int64 -> Int64 -> IO a)
  -> [Int64]
  -> IO (String, String)
showTensor get'1d get'2d get'3d get'4d ds =
  (,desc) <$> case ds of
    []  -> pure ""
    [x] -> brackets . intercalate "" <$> mapM (fmap valWithSpace . get'1d) (mkIx x)
    [x,y] -> go "" get'2d x y
    [x,y,z] -> mat3dGo x y z
    [x,y,z,q] -> mat4dGo x y z q
    _ -> pure "Can't print this yet"
 where
  go :: String -> (Int64 -> Int64 -> IO a) -> Int64 -> Int64 -> IO String
  go fill getter x y = do
    vs <- forM (mkXY x y) $ fmap valWithSpace . uncurry getter
    pure (mat2dGo fill y "" vs)

  mat2dGo :: String -> Int64 -> String -> [String] -> String
  mat2dGo    _ _ acc []  = acc
  mat2dGo fill y acc rcs = mat2dGo fill y acc' rest
    where
      (row, rest) = splitAt (fromIntegral y) rcs
      fullrow = fill ++ brackets (intercalate "" row)
      acc' = if null acc then fullrow else acc ++ "\n" ++ fullrow

  mat3dGo :: Int64 -> Int64 -> Int64 -> IO String
  mat3dGo x y z = fmap (intercalate "") $ forM (mkIx x) $ \x' -> do
    mat <- go "  " (get'3d x') y z
    pure $ gt2IxHeader [x'] ++ mat

  mat4dGo :: Int64 -> Int64 -> Int64 -> Int64 -> IO String
  mat4dGo w q x y = fmap (intercalate "") $ forM (mkXY w q) $ \(w', q') -> do
    mat <- go "  " (get'4d w' q') x y
    pure $ gt2IxHeader [w', q'] ++ mat

  gt2IxHeader :: [Int64] -> String
  gt2IxHeader is = "\n(" ++ intercalate "," (fmap show is) ++",.,.):\n"

  mkIx :: Int64 -> [Int64]
  mkIx x = [0..x - 1]

  mkXY :: Int64 -> Int64 -> [(Int64, Int64)]
  mkXY x y = [ (r, c) | r <- mkIx x, c <- mkIx y ]

  brackets :: String -> String
  brackets s = "[" ++ s ++ "]"

  valWithSpace :: (Typeable a, Ord a, Num a, Show a) => a -> String
  valWithSpace v = spacing ++ value ++ ""
   where
     truncTo :: (RealFrac x, Fractional x) => Int -> x -> x
     truncTo n f = fromInteger (round $ f * (10^n)) / (10.0^^n)

     value :: String
     value = fromMaybe (show v) $
           (printf "%.4f" <$> (cast v :: Maybe Double))
       <|> (printf "%.4f" <$> (cast v :: Maybe Float))

     spacing = magspacing ++ signspacing
     magspacing = ""
     -- magspacing = case compare (v `mod` 10) 4 of
     --   LT -> replicate (v `mod` 10)
     signspacing = case compare (signum v) 0 of
        LT -> " "
        _  -> "  "

  descType, descShape, desc :: String
  descType = show (typeRep (Proxy :: Proxy a)) ++ " tensor with "
  descShape = "shape: " ++ intercalate "x" (fmap show ds)
  desc = brackets $ descType ++ descShape

data TenSlices
  = TenNone
  | TenVector (NonEmpty HsReal)
  | TenMatricies (NonEmpty (NonEmpty [HsReal]))

-- | Helper function to show the matrix slices from a tensor.
tensorSlices
  :: Dynamic
  -> (Int64 -> IO HsReal)
  -> (Int64 -> Int64 -> IO HsReal)
  -- -> (Int64 -> Int64 -> Int64 -> IO HsReal)
  -- -> (Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal)
  -> [Word64]
  -> IO TenSlices
tensorSlices t get'1d get'2d -- get'3d get'4d
  = \case
    []  -> pure TenNone
    [x] -> TenVector <$> go1d get'1d x
    [x,y] -> (TenMatricies . (:|[])) <$> go2d get'2d x y
    _ -> throwString "Can't slice this yet"
 where
  go1d :: (Int64 -> IO HsReal) -> Word64 -> IO (NonEmpty HsReal)
  go1d getter x
    = forM (mkIx x) getter

  go2d :: (Int64 -> Int64 -> IO HsReal) -> Word64 -> Word64 -> IO (NonEmpty [HsReal])
  go2d getter x y =
    forM (mkIx x) $ \ix ->
      forM (mkVIx y) $ \iy ->
        getter ix iy

  go3d :: (Int64 -> Int64 -> Int64 -> IO HsReal) -> Word64 -> Word64 -> Word64 -> IO (NonEmpty (NonEmpty [HsReal]))
  go3d getter x y z =
    forM (mkIx x) $ \ix ->
      forM (mkIx y) $ \iy ->
        -- forM [0..z - 1] $ \iz ->
          traverse (getter ix iy) (mkVIx z)

  -- mat2dGo :: Int64 -> String -> [HsReal] -> String
  -- mat2dGo _ acc []  = acc
  -- mat2dGo y acc rcs = mat2dGo y acc' rest
  --   where
  --     (row, rest) = splitAt (fromIntegral y) rcs
  --     acc' = if null acc then row else acc ++ "\n" ++ row

  -- mat3dGo :: Int64 -> Int64 -> Int64 -> IO String
  -- mat3dGo x y z = fmap (intercalate "") $ forM (mkIx x) $ \x' -> do
  --   mat <- go "  " (get'3d x') y z
  --   pure $ gt2IxHeader [x'] ++ mat

  -- mat4dGo :: Int64 -> Int64 -> Int64 -> Int64 -> IO String
  -- mat4dGo w q x y = fmap (intercalate "") $ forM (mkXY w q) $ \(w', q') -> do
  --   mat <- go "  " (get'4d w' q') x y
  --   pure $ gt2IxHeader [w', q'] ++ mat

  mkIx :: Word64 -> NonEmpty Int64
  mkIx 0 = 0 :| []
  mkIx x = 0 :| [1..fromIntegral x - 1]

  mkVIx :: Word64 -> [Int64]
  mkVIx 0 = []
  mkVIx x = [0..fromIntegral x - 1]


-------------------------------------------------------------------------------
-- * Helper functions

-- | run a function with a dynamic tensor and storage's underlying implementation details.
withDynamicStateAndStorage :: Sig.Dynamic -> Sig.Storage -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CStorage -> IO x) -> IO x
withDynamicStateAndStorage t s fn = flip with pure $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ withForeignPtr (Sig.cstorage s) (fn s' t')

-- | exported helper function. Not actually "inplace" this is actually "with return and static dimensions"
withInplace :: (Dynamic -> IO ()) -> Dims (d::[Nat]) -> IO Dynamic
withInplace op d = new d >>= \r -> op r >> pure r

-- | exported helper function. not actually "inplace" this is actually "with return and runtime dimensions"
withInplace' :: (Dynamic -> IO ()) -> SomeDims -> IO Dynamic
withInplace' op (SomeDims d) = withInplace op d

-- | exported helper function. This is actually 'inplace'
twice :: Dynamic -> (Dynamic -> Dynamic -> IO ()) -> IO Dynamic
twice t op = op t t >> pure t

-- | exported helper function. Should be renamed to @newFromSize@
withEmpty :: Dynamic -> (Dynamic -> IO ()) -> IO Dynamic
withEmpty t op = getDims t >>= new' >>= \r -> op r >> pure r

-- | exported helper function. We can get away with this some of the time, when Torch
-- does the resizing in C, but you need to look at the c implementation
withEmpty' :: (Dynamic -> IO ()) -> IO Dynamic
withEmpty' op = empty >>= \r -> op r >> pure r

-- ========================================================================= --
-- Attempt at super-crazy tensor function

class NestableList t where
  fromNested :: t -> ExceptT String IO Dynamic

instance NestableList [HsReal] where
  fromNested = vector

instance NestableList [[HsReal]] where
  fromNested = matrix

instance NestableList [[[HsReal]]] where
  fromNested = cuboid

instance NestableList [[[[HsReal]]]] where
  fromNested = hyper


