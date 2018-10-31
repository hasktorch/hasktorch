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
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fno-cse -Wno-deprecations #-} -- no deprications because we still bundle up all mutable functions
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
import Data.Maybe (fromMaybe, fromJust)
import Data.List (intercalate, genericLength)
import Data.List.NonEmpty (NonEmpty(..))
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import GHC.Exts (IsList(..))
import Numeric.Dimensions
import System.IO.Unsafe
import Control.Concurrent
import Control.Monad.Trans.Except
import Text.Printf

import qualified Data.List                 as List ((!!))
import qualified Data.List.NonEmpty        as NE
import qualified Torch.Types.TH            as TH
import qualified Foreign.Marshal.Array     as FM
import qualified Torch.Sig.State           as Sig
import qualified Torch.Sig.Types           as Sig
import qualified Torch.Sig.Types.Global    as Sig
import qualified Torch.Sig.Tensor          as Sig
import qualified Torch.Sig.Tensor.Memory   as Sig
import qualified Torch.Sig.Storage         as StorageSig (c_size)

import Torch.Indef.Dynamic.Print (showTensor, describeTensor)
import Torch.Indef.Types
import Torch.Indef.Internal
import Torch.Indef.Index hiding (withDynamicState)
import qualified Torch.Indef.Storage as Storage

-- | Clears the internal flags on a tensor. Uses bitwise operators for flags.
_clearFlag :: Dynamic -> Int8 -> IO ()
_clearFlag t cc = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_clearFlag s' t' (CChar cc)

-- | Get the underlying data as a haskell list from the tensor
--
-- NOTE: This _cannot_ use a Tensor's storage size because ATen's Storage
-- allocates up to the next 64-byte line on the CPU (needs reference, this
-- is the unofficial response from \@soumith in slack).
tensordata :: Dynamic -> [HsReal]
tensordata t =
  case shape t of
    [] -> []
    ds ->
      unsafeDupablePerformIO . flip with (pure . fmap c2hsReal) $ do
        st <- managedState
        t' <- managedTensor t
        liftIO $ do
          let sz = fromIntegral (product ds)
          -- a strong dose of paranoia
          tmp <- FM.mallocArray sz
          creals <- Sig.c_data st t'
          FM.copyArray tmp creals sz
          FM.peekArray sz tmp
{-# NOINLINE tensordata #-}

-- | get a value from dimension 1
get1d :: Dynamic -> Word -> Maybe HsReal
get1d t d1
  | nDimension t /= 1 || size t 0 < d1 = Nothing
  | otherwise = unsafeDupablePerformIO . flip with (pure . Just . c2hsReal) . (liftIO =<<) $ Sig.c_get1d
    <$> managedState
    <*> managedTensor t
    <*> pure (fromIntegral d1)
{-# NOINLINE get1d #-}

unsafeGet1d :: Dynamic -> Word -> HsReal
unsafeGet1d t d1 = fromJust $ get1d t d1

-- | get a value from dimension 2
get2d :: Dynamic -> Word -> Word -> Maybe HsReal
get2d t d1 d2
  | nDimension t /= 2 = Nothing
  | otherwise = unsafeDupablePerformIO . flip with (pure . Just . c2hsReal) . (liftIO =<<) $ Sig.c_get2d
    <$> managedState
    <*> managedTensor t
    <*> pure (fromIntegral d1)
    <*> pure (fromIntegral d2)
{-# NOINLINE get2d #-}

unsafeGet2d :: Dynamic -> Word -> Word -> HsReal
unsafeGet2d t d1 d2 = fromJust $ get2d t d1 d2

-- | get a value from dimension 3
get3d :: Dynamic -> Word -> Word -> Word -> Maybe HsReal
get3d t d1 d2 d3
  | nDimension t /= 3 = Nothing
  | otherwise = unsafeDupablePerformIO . flip with (pure . Just . c2hsReal) . (liftIO =<<) $ Sig.c_get3d
    <$> managedState
    <*> managedTensor t
    <*> pure (fromIntegral d1)
    <*> pure (fromIntegral d2)
    <*> pure (fromIntegral d3)
{-# NOINLINE get3d #-}

unsafeGet3d :: Dynamic -> Word -> Word -> Word -> HsReal
unsafeGet3d t d1 d2 d3 = fromJust $ get3d t d1 d2 d3

-- | get a value from dimension 4
get4d :: Dynamic -> Word -> Word -> Word -> Word -> Maybe HsReal
get4d t d1 d2 d3 d4
  | nDimension t /= 4 = Nothing
  | otherwise = unsafeDupablePerformIO . flip with (pure . Just . c2hsReal) . (liftIO =<<) $ Sig.c_get4d
    <$> managedState
    <*> managedTensor t
    <*> pure (fromIntegral d1)
    <*> pure (fromIntegral d2)
    <*> pure (fromIntegral d3)
    <*> pure (fromIntegral d4)
{-# NOINLINE get4d #-}

unsafeGet4d :: Dynamic -> Word -> Word -> Word -> Word -> HsReal
unsafeGet4d t d1 d2 d3 d4 = fromJust $ get4d t d1 d2 d3 d4

-- | get a value from a dynamic tensor at a given index, @Dims d@.
getDim :: Dynamic -> Dims ((i:+ds)::[Nat]) -> Maybe HsReal
getDim t d = case fromIntegral <$> listDims d of
  []           -> error "[impossible] pattern match fail, `Dims ((i:+ds)::[Nat])` prevents this"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> error "[incomplete] getDim doen't have support for dimensions > 4"

-- | whether or not the tensor is contiguous in memory.
isContiguous :: Dynamic -> Bool
isContiguous t = unsafeDupablePerformIO . flip with (pure . (1 ==)) . (liftIO =<<) $ Sig.c_isContiguous
  <$> managedState
  <*> managedTensor t
{-# NOINLINE isContiguous #-}

-- | check to see if to tensors are the same size as eachother.
isSameSizeAs :: Dynamic -> Dynamic -> Bool
isSameSizeAs t0 t1 = unsafeDupablePerformIO . flip with (pure . (1 ==)) . (liftIO =<<) $ Sig.c_isSameSizeAs
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
{-# NOINLINE isSameSizeAs #-}

-- | Returns true iff the Tensor is set to the argument Tensor.
--
-- Note: this is only true if the tensors are the same size, have
-- the same strides and share the same storage and offset.
isSetTo :: Dynamic -> Dynamic -> Bool
isSetTo t0 t1 = unsafeDupablePerformIO . flip with (pure . (1 ==)) . (liftIO =<<) $ Sig.c_isSetTo
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
{-# NOINLINE isSetTo #-}

-- | check to see if the tensor is the same size as the LongStorage.
isSize :: Dynamic -> TH.LongStorage -> Bool
isSize t ls = unsafeDupablePerformIO . flip with (pure . (1 ==)) . (liftIO =<<) $ Sig.c_isSize
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (snd $ TH.longStorageState ls))
{-# NOINLINE isSize #-}

-- | Returns the number of dimensions in a Tensor.
nDimension :: Dynamic -> Word
nDimension t = unsafeDupablePerformIO . flip with (pure . fromIntegral) . (liftIO =<<) $ Sig.c_nDimension
  <$> managedState
  <*> managedTensor t
{-# NOINLINE nDimension #-}

-- | Returns the number of elements in a Tensor.
nElement :: Dynamic -> Word64
nElement t = unsafeDupablePerformIO . flip with (pure . fromIntegral) . (liftIO =<<) $ Sig.c_nElement
  <$> managedState
  <*> managedTensor t
{-# NOINLINE nElement #-}

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
_narrow
  :: Dynamic -- ^ return tensor to mutate (C-style)
  -> Dynamic -- ^ source tensor used for data
  -> Word    -- ^ dimension to operate on
  -> Int64
  -> Size
  -> IO ()
_narrow t0 t1 a b c = withLift $ Sig.c_narrow
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
  <*> pure (fromIntegral c)
{-# WARNING _narrow "hasktorch devs have not yet made this safe. You are warned." #-}

-- | Returns an empty tensor.
empty :: Dynamic
empty = unsafeDupablePerformIO . withDynamic $ Sig.c_new <$> managedState
{-# NOINLINE empty #-}

-- | pure version of '_expand'
newExpand :: Dynamic -> TH.IndexStorage -> Dynamic
newExpand r ix = unsafeDupablePerformIO . withDynamic $ Sig.c_newExpand
  <$> managedState
  <*> managedTensor r
  <*> managed (withForeignPtr . snd $ TH.longStorageState ix)
{-# NOINLINE newExpand #-}

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
_expand r t ix = withLift $ Sig.c_expand
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
  mngNonEmpty = mapM toMPtr . NE.toList >=> mWithArray

  mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
  mWithArray as = managed (FM.withArray as)

  toMPtr :: Dynamic -> Managed (Ptr CTensor)
  toMPtr d = managed (withForeignPtr (Sig.ctensor d))

-- | purely clone a tensor
newClone :: Dynamic -> Dynamic
newClone t = unsafeDupablePerformIO . withDynamic $ Sig.c_newClone
  <$> managedState
  <*> managedTensor t
{-# NOINLINE newClone #-}

-- | purely clone a tensor to have a contiguous memory layout.
newContiguous :: Dynamic -> Dynamic
newContiguous t = unsafeDupablePerformIO . withDynamic $ Sig.c_newContiguous
  <$> managedState
  <*> managedTensor t
{-# NOINLINE newContiguous #-}

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
newNarrow
  :: Dynamic    -- ^ source tensor
  -> Word       -- ^ dimenion to operate over
  -> Int64
  -> Size
  -> IO Dynamic -- ^ return tensor, linked by storage to source tensor
newNarrow t a b c = withDynamic $ Sig.c_newNarrow
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
  <*> pure (fromIntegral c)
{-# WARNING newNarrow "hasktorch devs have not yet made this safe. You are warned." #-}

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
newSelect
  :: Dynamic    -- ^ source tensor
  -> Word       -- ^ dimension to operate over
  -> Int64
  -> IO Dynamic -- ^ return tensor, linked by storage to source tensor
newSelect t a b = withDynamic $ Sig.c_newSelect
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
{-# WARNING newSelect "hasktorch devs have not yet made this safe. You are warned." #-}

-- | get the sizes of each dimension
--
-- FIXME: doublecheck this
newSizeOf :: Dynamic -> TH.IndexStorage
newSizeOf t = unsafeDupablePerformIO . flip with mkCPUIxStorage $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newSizeOf s' t'
{-# NOINLINE newSizeOf #-}

-- | get the strides of each dimension
--
-- FIXME: doublecheck this
newStrideOf :: Dynamic -> TH.IndexStorage
newStrideOf t = unsafeDupablePerformIO . flip with mkCPUIxStorage $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_newStrideOf s' t'
{-# NOINLINE newStrideOf #-}

-- | pure version of '_transpose'
newTranspose :: Dynamic -> Word -> Word -> Dynamic
newTranspose t a b = unsafeDupablePerformIO . withDynamic $ Sig.c_newTranspose
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
{-# NOINLINE newTranspose #-}

-- | pure version of '_unfold'
newUnfold
  :: Dynamic  -- ^ source tensor
  -> Word     -- ^ dimension to operate on
  -> Int64
  -> Int64
  -> Dynamic  -- ^ return tensor
newUnfold t a b c = unsafeDupablePerformIO . withDynamic $ Sig.c_newUnfold
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
  <*> pure (fromIntegral c)
{-# NOINLINE newUnfold #-}

-- |
-- Creates a view with different dimensions of the storage associated with tensor, returning a new tensor.
--
-- FIXME: I think resizeAs is the non-cloning version of this function. See:
-- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes
--
-- for more.
--
-- NOTE(stites): I think this API can only be kept pure via linear types.
newView :: Dynamic -> TH.IndexStorage -> IO Dynamic
newView t ix = withDynamic $ Sig.c_newView
  <$> managedState
  <*> managedTensor t
  <*> managed (withCPUIxStorage ix)
{-# WARNING newView "hasktorch devs have not yet made this safe. You are warned." #-}

-- | create an uninitialized tensor with the given size and strides (?)
--
-- FIXME: doublecheck what the IndexStorages stands for
newWithSize :: TH.IndexStorage -> TH.IndexStorage -> Dynamic
newWithSize l0 l1 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithSize
  <$> managedState
  <*> managed (withCPUIxStorage l0)
  <*> managed (withCPUIxStorage l1)
{-# NOINLINE newWithSize #-}

-- | create an uninitialized 1d tensor
newWithSize1d :: Word -> Dynamic
newWithSize1d a0 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithSize1d
  <$> managedState
  <*> pure (fromIntegral a0)
{-# NOINLINE newWithSize1d #-}

-- | create an uninitialized 2d tensor
newWithSize2d :: Word -> Word -> Dynamic
newWithSize2d a0 a1 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithSize2d
  <$> managedState
  <*> pure (fromIntegral a0)
  <*> pure (fromIntegral a1)
{-# NOINLINE newWithSize2d #-}

-- | create an uninitialized 3d tensor
newWithSize3d :: Word -> Word -> Word -> Dynamic
newWithSize3d a0 a1 a2 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithSize3d
  <$> managedState
  <*> pure (fromIntegral a0)
  <*> pure (fromIntegral a1)
  <*> pure (fromIntegral a2)
{-# NOINLINE newWithSize3d #-}

-- | create an uninitialized 4d tensor
newWithSize4d :: Word -> Word -> Word -> Word -> Dynamic
newWithSize4d a0 a1 a2 a3 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithSize4d
  <$> managedState
  <*> pure (fromIntegral a0)
  <*> pure (fromIntegral a1)
  <*> pure (fromIntegral a2)
  <*> pure (fromIntegral a3)
{-# NOINLINE newWithSize4d #-}

-- | create a new tensor with the given size and strides, storage offset and storage.
newWithStorage :: Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> Dynamic
newWithStorage s pd l0 l1 = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithStorage
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> managed (withForeignPtr (snd $ TH.longStorageState l0))
  <*> managed (withForeignPtr (snd $ TH.longStorageState l1))
{-# NOINLINE newWithStorage #-}


-- | create a new 1d tensor with the given storage's first dimension.
newWithStorage1d
  :: Storage            -- storage to use
  -> StorageOffset      -- storage offset must be >= 1
  -> (Size, Stride)     -- size is of the 1st dimension, stride is the stride in the first dimension
  -> Dynamic
newWithStorage1d s pd (d00,d01) = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithStorage1d
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
{-# NOINLINE newWithStorage1d #-}


-- | create a new 2d tensor with the given storage's first 2 dimensions.
newWithStorage2d
  :: Storage
  -> StorageOffset
  -> (Size, Stride)
  -> (Size, Stride)
  -> Dynamic
newWithStorage2d s pd (d00,d01) (d10,d11) = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithStorage2d
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
{-# NOINLINE newWithStorage2d #-}


-- | create a new 3d tensor with the given storage's first 3 dimensions.
newWithStorage3d
  :: Storage
  -> StorageOffset
  -> (Size, Stride)
  -> (Size, Stride)
  -> (Size, Stride)
  -> Dynamic
newWithStorage3d s pd (d00,d01) (d10,d11) (d20,d21) = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithStorage3d
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
  <*> pure (fromIntegral d20) <*> pure (fromIntegral d21)
{-# NOINLINE newWithStorage3d #-}


-- | create a new 4d tensor with the given storage's first 4 dimensions.
newWithStorage4d
  :: Storage
  -> StorageOffset
  -> (Size, Stride)
  -> (Size, Stride)
  -> (Size, Stride)
  -> (Size, Stride)
  -> Dynamic
newWithStorage4d s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) = unsafeDupablePerformIO . withDynamic $ Sig.c_newWithStorage4d
  <$> managedState
  <*> managedStorage s
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
  <*> pure (fromIntegral d20) <*> pure (fromIntegral d21)
  <*> pure (fromIntegral d30) <*> pure (fromIntegral d31)
{-# NOINLINE newWithStorage4d #-}

-- | create a new tensor with the given tensor's underlying storage.
newWithTensor :: Dynamic -> IO Dynamic
newWithTensor t = withDynamic $ Sig.c_newWithTensor
  <$> managedState
  <*> managedTensor t
{-# NOINLINE newWithTensor #-}
{-# WARNING newWithTensor "this function causes the input tensor to be impure" #-}

-- | Resize the tensor according to the given LongStorage size (and strides?)
-- FIXME: doublecheck what the IndexStorages stands for
_resize
  :: Dynamic -> TH.IndexStorage -> TH.IndexStorage -> IO ()
_resize t l0 l1 = withLift $ Sig.c_resize
  <$> managedState
  <*> managedTensor t
  <*> managed (withCPUIxStorage l0)
  <*> managed (withCPUIxStorage l1)

-- | resize dimension 1 of a tensor.
resize1d_ :: Dynamic -> Word -> IO ()
resize1d_ t l0 = withLift $ Sig.c_resize1d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)

-- | resize the first 2 dimensions of a tensor.
resize2d_ :: Dynamic -> Word -> Word -> IO ()
resize2d_ t l0 l1 = withLift $ Sig.c_resize2d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)

-- | resize the first 3 dimensions of a tensor.
resize3d_ :: Dynamic -> Word -> Word -> Word -> IO ()
resize3d_ t l0 l1 l2 = withLift $ Sig.c_resize3d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (fromIntegral l2)

-- | resize the first 4 dimensions of a tensor.
resize4d_ :: Dynamic -> Word -> Word -> Word -> Word -> IO ()
resize4d_ t l0 l1 l2 l3 = withLift $ Sig.c_resize4d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (fromIntegral l2)
  <*> pure (fromIntegral l3)

-- | resize the first 5 dimensions of a tensor.
resize5d_ :: Dynamic -> Word -> Word -> Word -> Word -> Word -> IO ()
resize5d_ t l0 l1 l2 l3 l4 = withLift $ Sig.c_resize5d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (fromIntegral l2)
  <*> pure (fromIntegral l3)
  <*> pure (fromIntegral l4)

-- | Resize the tensor as the given tensor.
resizeAs_
  :: Dynamic -- ^ tensor to mutate inplace
  -> Dynamic -- ^ tensor used for its shape
  -> IO ()
resizeAs_ t0 t1 = with2DynamicState t0 t1 Sig.c_resizeAs

-- | resize a tensor with given strides, sizes and a magical parameter.
--
-- FIXME: Someone needs to find out what the magical parameter is.
resizeNd_
  :: Dynamic   -- ^ tensor to resize inplace.
  -> Int32     -- ^ unknown argument. FIXME: Someone needs to find this out.
  -> [Size]    -- ^ new sizes to update
  -> [Stride]  -- ^ new strides to update.
  -> IO ()
resizeNd_ t i l0' l1' = withLift $ Sig.c_resizeNd
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral i)
  <*> liftIO (FM.newArray (coerce l0' :: [CLLong]))
  <*> liftIO (FM.newArray (coerce l1' :: [CLLong]))

-- | Increment the reference counter of the tensor.
--
-- From: https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/tensor.md#reference-counting
--
-- Tensors are reference-counted. It means that each time an object (C or the Lua state) need to keep a reference over a tensor, the corresponding tensor reference counter will be increased. The reference counter is decreased when the object does not need the tensor anymore.
--
-- These methods should be used with extreme care. In general, they should never be called, except if you know what you are doing, as the handling of references is done automatically. They can be useful in threaded environments. Note that these methods are atomic operations.
retain :: Dynamic -> IO ()
retain t = withLift $ Sig.c_retain
  <$> managedState
  <*> managedTensor t

-- | returns a tensor which /shares the same 'Storage'/ as the original. Hence, any modification in
-- the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.
-- These methods are very fast, as they do not involve any memory copy.
_select
  :: Dynamic  -- ^ return tensor which is mutated inplace (C-Style)
  -> Dynamic  -- ^ source tensor
  -> Word     -- ^ dimension to operate over
  -> Word
  -> IO ()
_select t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_select s' t0' t1' (fromIntegral a) (fromIntegral b)
{-# WARNING _select "hasktorch devs have not yet made this safe. You are warned." #-}

-- | set the source tensor's storage to another tensor.
_set
  :: Dynamic  -- ^ the source tensor which is mutated inplace
  -> Dynamic  -- ^ the tensor who's storage is going to be referenced.
  -> IO ()
_set t0 t1 = with2DynamicState t0 t1 Sig.c_set
{-# WARNING _set "hasktorch devs have not yet made this safe. You are warned." #-}

-- | set a value in dimension 1, inplace.
set1d_
  :: Dynamic -- ^ source tensor
  -> Word    -- ^ rank-1 index
  -> HsReal  -- ^ value to put
  -> IO ()
set1d_ t l0 v = withLift $ Sig.c_set1d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (hs2cReal v)

-- | set a value in dimension 2, inplace.
set2d_
  :: Dynamic -- ^ source tensor
  -> Word    -- ^ rank-1 index
  -> Word    -- ^ rank-2 index
  -> HsReal  -- ^ value to put
  -> IO ()
set2d_ t l0 l1 v = withLift $ Sig.c_set2d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (hs2cReal v)

-- | set a value in dimension 3, inplace.
set3d_
  :: Dynamic -- ^ source tensor
  -> Word    -- ^ rank-1 index
  -> Word    -- ^ rank-2 index
  -> Word    -- ^ rank-3 index
  -> HsReal  -- ^ value to put
  -> IO ()
set3d_ t l0 l1 l2 v = withLift $ Sig.c_set3d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (fromIntegral l2)
  <*> pure (hs2cReal v)

-- | set a value in dimension 4, inplace.
set4d_
  :: Dynamic -- ^ source tensor
  -> Word    -- ^ rank-1 index
  -> Word    -- ^ rank-2 index
  -> Word    -- ^ rank-3 index
  -> Word    -- ^ rank-4 index
  -> HsReal  -- ^ value to put
  -> IO ()
set4d_ t l0 l1 l2 l3 v = withLift $ Sig.c_set4d
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral l0)
  <*> pure (fromIntegral l1)
  <*> pure (fromIntegral l2)
  <*> pure (fromIntegral l3)
  <*> pure (hs2cReal v)

-- | set the flags on a tensor inplace
setFlag_ :: Dynamic -> Int8 -> IO ()
setFlag_ t l0 = withLift $ Sig.c_setFlag
  <$> managedState
  <*> managedTensor t
  <*> pure (CChar l0)

-- | Set the storage of a tensor.
--
-- FIXME: doublecheck what the IndexStorages stands for
setStorage_ :: Dynamic -> Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO ()
setStorage_ t s a b c = withLift $ Sig.c_setStorage
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral a)
  <*> managed (withCPUIxStorage b)
  <*> managed (withCPUIxStorage c)
{-# WARNING setStorage_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | Set the storage of a tensor, only referencing 1 dimension of storage
setStorage1d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
setStorage1d_ t s pd (d00,d01) = withLift $ Sig.c_setStorage1d
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
{-# WARNING setStorage1d_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | Set the storage of a tensor, only referencing 2 dimensions of storage
setStorage2d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
setStorage2d_ t s pd (d00,d01) (d10,d11) = withLift $ Sig.c_setStorage2d
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
{-# WARNING setStorage2d_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}


-- | Set the storage of a tensor, only referencing 3 dimensions of storage
setStorage3d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
setStorage3d_ t s pd (d00,d01) (d10,d11) (d20,d21) = withLift $ Sig.c_setStorage3d
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
  <*> pure (fromIntegral d20) <*> pure (fromIntegral d21)
{-# WARNING setStorage3d_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | Set the storage of a tensor, only referencing 4 dimensions of storage
setStorage4d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
setStorage4d_ t s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) = withLift $ Sig.c_setStorage4d
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral pd)
  <*> pure (fromIntegral d00) <*> pure (fromIntegral d01)
  <*> pure (fromIntegral d10) <*> pure (fromIntegral d11)
  <*> pure (fromIntegral d20) <*> pure (fromIntegral d21)
  <*> pure (fromIntegral d30) <*> pure (fromIntegral d31)
{-# WARNING setStorage4d_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | Set the storage of a tensor, referencing any number of dimensions of storage
setStorageNd_
  :: Dynamic       -- ^ tensor to mutate, inplace
  -> Storage       -- ^ storage to set
  -> StorageOffset -- ^ offset of the storage to start from
  -> Word          -- ^ dimension... to operate over? to start from? (TODO: allow for "unset" dimension)
  -> [Size]        -- ^ sizes to use with the storage
  -> [Stride]      -- ^ strides to use with the storage
  -> IO ()
setStorageNd_ t s a b hsc hsd = withLift $ Sig.c_setStorageNd
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
  <*> liftIO (FM.newArray (coerce hsc :: [CLLong]))
  <*> liftIO (FM.newArray (coerce hsd :: [CLLong]))
{-# WARNING setStorageNd_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | get the size of a tensor's specific dimension.
--
-- FIXME: this can throw an exception if the dimension is out-of-bound.
size
  :: Dynamic -- ^ tensor to inspect
  -> Word    -- ^ dimension to get
  -> Word    -- ^ size of the dimension
size t d = unsafeDupablePerformIO . flip with (pure . fromIntegral) . (liftIO =<<) $ Sig.c_size
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral d)
  -- liftIO $  s' t' (fromIntegral l0)

-- | primarily used for debugging. Get the size description from a c call.
sizeDesc :: Dynamic -> IO DescBuff
sizeDesc t = flip with (Sig.descBuff) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_sizeDesc s' t'

-- | Removes all singleton dimensions of the tensor.
_squeeze :: Dynamic -> Dynamic -> IO ()
_squeeze t0 t1 = withLift $ Sig.c_squeeze
  <$> managedState
  <*> managedTensor t1
  <*> managedTensor t0

-- | Removes a singleton dimensions of the tensor at a given dimension.
squeeze1d_
  :: Dynamic -- ^ tensor to mutate
  -> Word    -- ^ dimension to squeeze
  -> IO ()
squeeze1d_ t d = _squeeze1d t t d

-- | Removes a singleton dimensions of the tensor at a given dimension.
_squeeze1d
  :: Dynamic -- ^ tensor to mutate as return (C-Style)
  -> Dynamic -- ^ source tensor
  -> Word    -- ^ dimension to squeeze
  -> IO ()
_squeeze1d t0 t1 d = withLift $ Sig.c_squeeze1d
  <$> managedState
  <*> managedTensor t1
  <*> managedTensor t0
  <*> pure (fromIntegral d)

-- | get the underlying storage of a tensor
storage :: Dynamic -> Storage
storage t = unsafeDupablePerformIO . withStorage $ Sig.c_storage
  <$> managedState
  <*> managedTensor t
{-# NOINLINE storage #-}
{-# WARNING storage "extracting and using a tensor's storage can make your program unsafe. You are warned." #-}

-- | get the storage offset of a tensor
storageOffset :: Dynamic -> StorageOffset
storageOffset t = fromIntegral . unsafeDupablePerformIO . withLift $ Sig.c_storageOffset
  <$> managedState
  <*> managedTensor t
{-# NOINLINE storageOffset #-}

-- | Returns the jump necessary to go from one element to the next one in the
-- specified dimension dim.
stride
  :: Dynamic   -- ^ tensor to query
  -> Word      -- ^ dimension of tensor
  -> IO Stride -- ^ stride of dimension
stride t a = flip with (pure . fromIntegral) . (liftIO =<<) $ Sig.c_stride
  <$> managedState
  <*> managedTensor t
  <*> pure (fromIntegral a)

-- | Returns a tensor where dimensions dim1 and dim2 have been swapped.
_transpose
  :: Dynamic  -- ^ tensor to mutate into the result.
  -> Dynamic  -- ^ source tensor to use for data.
  -> Word   -- ^ dim1
  -> Word   -- ^ dim2
  -> IO ()
_transpose t0 t1 a b = withLift $ Sig.c_transpose
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)

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
  -> Word     -- ^ dimension to operate on
  -> Size
  -> Step
  -> IO ()
_unfold t0 t1 a b c = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_unfold s' t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

-- | unsqueeze a tensor inplace
unsqueeze1d_
  :: Dynamic  -- ^ tensor to mutate
  -> Word     -- ^ dimension to unsqueeze
  -> IO ()
unsqueeze1d_ t = _unsqueeze1d t t

-- | unsqueeze a tensor, adding a singleton dimension at the specified dimval.
_unsqueeze1d
  :: Dynamic  -- ^ tensor to mutate into the result.
  -> Dynamic  -- ^ source tensor to use for data.
  -> Word     -- ^ dimension to unsqueeze
  -> IO ()
_unsqueeze1d t0 t1 d = withLift $ Sig.c_unsqueeze1d
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral d)

-- ========================================================================= --
-- User API (can be bundled into the above)
-- ========================================================================= --

-- | return the a runtime shape representing the dimensions of a 'Dynamic'
shape :: Dynamic -> [Word]
shape t = case nDimension t of
  0 -> []
  d -> (size t . fromIntegral) <$> [0.. d - 1]

-- | set the storage dimensionality of a dynamic tensor, inplace, to any new size and stride pair.
setStorageDim_ :: Dynamic -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
setStorageDim_ t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> setStorage1d_ t s o x
  [x, y]       -> setStorage2d_ t s o x y
  [x, y, z]    -> setStorage3d_ t s o x y z
  [x, y, z, q] -> setStorage4d_ t s o x y z q
  _            -> throwGT4 "setStorage"
{-# WARNING setStorageDim_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | set a value of a dynamic tensor, inplace, with any dimensionality.
setDim_ :: Dynamic -> Dims (d::[Nat]) -> HsReal -> IO ()
setDim_ t d !v = do
 threadDelay 1000
 case fromIntegral <$> listDims d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> set1d_ t x       v
  [x, y]       -> set2d_ t x y     v
  [x, y, z]    -> set3d_ t x y z   v
  [x, y, z, q] -> set4d_ t x y z q v
  _            -> throwGT4 "set"

-- | resize a dynamic tensor, inplace, to any new dimensionality
resizeDim_ :: Dynamic -> Dims (d::[Nat]) -> IO ()
resizeDim_ t d = case fromIntegral <$> listDims d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> resize1d_ t x
  [x, y]          -> resize2d_ t x y
  [x, y, z]       -> resize3d_ t x y z
  [x, y, z, q]    -> resize4d_ t x y z q
  [x, y, z, q, w] -> resize5d_ t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

-- | create a 1d Dynamic tensor from a list of elements.
--
-- FIXME construct this with TH, not by using 'setDim' inplace (one-by-one) which might be doing a second linear pass.
-- FIXME: CUDA doesn't like the storage allocation:

vectorEIO :: [HsReal] -> ExceptT String IO Dynamic
vectorEIO l = lift $ do
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
  -- let st = (Storage.fromList l) -- (deepseq l l)
  pure $ newWithStorage1d (fromList l) 0 (genericLength l, 1)
-- #endif

vectorE :: [HsReal] -> Either String Dynamic
vectorE = unsafePerformIO . runExceptT . vectorEIO
{-# NOINLINE vectorE #-}

vector :: [HsReal] -> Maybe Dynamic
vector = either (const Nothing) Just . vectorE

-- | create a 2d Dynamic tensor from a list of list of elements.
matrix :: [[HsReal]] -> ExceptT String IO Dynamic
matrix ls
  | null ls = lift (pure empty)
  | any ((ncols /=) . length) ls = ExceptT . pure $ Left "rows are not all the same length"
  | otherwise = do
-- #ifdef HASKTORCH_INTERNAL_CUDA
    -- vec <- vector (deepseq l l)
    -- lift $ go vec (someDimsVal [nrows, ncols])
    -- pure vec
-- #else
    lift $ do
      -- st <- Storage.fromList (deepseq l l)
      pure $ newWithStorage2d (fromList l) 0 (nrows, ncols) (ncols, 1)
-- #endif
 where
  l = concat ls
  go vec (SomeDims ds) = resizeDim_ vec ds

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


-- | create a 3d Dynamic tensor (ie: rectangular cuboid) from a nested list of elements.
{-# NOINLINE cuboid #-}
cuboid :: [[[HsReal]]] -> ExceptT String IO Dynamic
cuboid ls
  | isEmpty ls = lift (pure empty)
  | null ls || any null ls || any (any null) ls
                                   = ExceptT . pure . Left $ "can't accept empty lists"
  | innerDimCheck ncols        ls  = ExceptT . pure . Left $ "rows are not all the same length"
  | innerDimCheck ndepth (head ls) = ExceptT . pure . Left $ "columns are not all the same length"

  | otherwise = lift $ do
      -- st <- Storage.fromList l
      -- FIXME: test that this is correct.
      pure $ newWithStorage3d (fromList l) 0 (nrows, ncols * ndepth) (ncols, ndepth) (ndepth, 1)
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
  go vec (SomeDims ds) = resizeDim_ vec ds >> pure vec

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
  | isEmpty ls = lift (pure empty)
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
      -- st <- Storage.fromList (deepseq l l)
      pure $ newWithStorage4d (fromList l) 0
        (nrows, ncols * ndepth * ntime)
        (ncols, ndepth * ntime)
        (ndepth, ntime)
        (ntime, 1)
-- #endif
 where
  l = concat (concat (concat ls))
  go vec (SomeDims ds) = resizeDim_ vec ds >> pure vec

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


-- -- | resize a dynamic tensor with runtime 'SomeDims' representation of its new shape. Returns a pure copy of the
-- -- input tensor.
-- resizeDim :: Dynamic -> Dims (d::[Nat]) -> IO Dynamic
-- resizeDim src d = let res = newClone src in resizeDim_ res d >> pure res
--
-- -- | alias to 'resizeDim' with a runtime 'SomeDims'
-- resizeDim' :: Dynamic -> SomeDims -> IO Dynamic
-- resizeDim' t (SomeDims d) = resizeDim t d

-- | get the runtime dimension list of a dynamic tensor
getDimsList :: Integral i => Dynamic -> [i]
getDimsList t = map (fromIntegral . size t) [0 .. nDimension t - 1]

-- | alias to 'getDimList' which wraps the dimensions list in a 'SomeDims'
getSomeDims :: Dynamic -> SomeDims
getSomeDims = someDimsVal . getDimsList

-- | create a new dynamic tensor of size @Dims d@
new :: Dims (d::[Nat]) -> Dynamic
new d = case fromIntegral <$> listDims d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> unsafeDupablePerformIO $ do
    let t = empty
    resizeDim_ t d
    pure t
{-# NOINLINE new #-}

-- | set a specific runtime 'SomeDims' position of a dynamic tensor.
setDim'_ :: Dynamic -> SomeDims -> HsReal -> IO ()
setDim'_ t (SomeDims d) v = setDim_ t d v

-- | resize a dynamic tensor inplace with runtime 'SomeDims' representation of its new shape.
resizeDim'_ :: Dynamic -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = resizeDim_ t d

-- | build a new tensor with a runtime 'SomeDims'
new' :: SomeDims -> Dynamic
new' (SomeDims d) = new d

-- | resize a tensor to take the shape of the second 'Dynamic' argument.
-- This is a pure function.
resizeAs
  :: Dynamic -- ^ src tensor to mutate
  -> Dynamic -- ^ a tensor only used for its shape
  -> IO Dynamic  -- ^ a new copy of src with the shape tensor's shape
resizeAs src shape = do
  let res = newClone src
  resizeAs_ res shape
  pure res

-------------------------------------------------------------------------------
-- * Helper functions

-- | run a function with a dynamic tensor and storage's underlying implementation details.
-- withDynamicStateAndStorage :: Sig.Dynamic -> Sig.Storage -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CStorage -> IO x) -> IO x
-- withDynamicStateAndStorage t s fn = flip with pure $ do
--   s' <- managedState
--   t' <- managedTensor t
--   liftIO $ withForeignPtr (Sig.cstorage s) (fn s' t')

-- | exported helper function. Not actually "inplace" this is actually "with return and static dimensions"
withInplace :: (Dynamic -> IO ()) -> Dims (d::[Nat]) -> IO Dynamic
withInplace op d =
  let
    r = new d
  in op r >> pure r

-- | exported helper function. not actually "inplace" this is actually "with return and runtime dimensions"
withInplace' :: (Dynamic -> IO ()) -> SomeDims -> IO Dynamic
withInplace' op (SomeDims d) = withInplace op d

-- | exported helper function. This is actually 'inplace'
twice :: Dynamic -> (Dynamic -> Dynamic -> IO ()) -> IO Dynamic
twice t op = op t t >> pure t

-- | exported helper function. Should be renamed to @newFromSize@
-- withEmpty :: Dynamic -> (Dynamic -> IO ()) -> IO Dynamic
-- withEmpty t op = let r = new' (getSomeDimsList t) in op r >> pure r

-- | exported helper function. We can get away with this some of the time, when Torch
-- does the resizing in C, but you need to look at the c implementation
withEmpty' :: (Dynamic -> IO ()) -> IO Dynamic
withEmpty' op = let r = empty in op r >> pure r


instance IsList Dynamic where
  type Item Dynamic = HsReal
  toList = tensordata
  fromList l = newWithStorage1d (fromList l) 0 (genericLength l, 1)

instance Show Dynamic where
  show t = vs ++ "\n" ++ desc
   where
    dims = getDimsList t
    desc = describeTensor dims (Proxy @HsReal)
    vs = showTensor
      (unsafeGet1d t)
      (unsafeGet2d t)
      (unsafeGet3d t)
      (unsafeGet4d t)
      dims

-- CPU ONLY:
--   desc :: Dynamic -> IO (DescBuff t)


