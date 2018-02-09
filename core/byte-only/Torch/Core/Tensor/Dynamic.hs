{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic
  ( ByteTensor
  -- , ShortTensor
  -- , IntTensor
  -- , LongTensor
  -- , FloatTensor
  -- , DoubleTensor

  , IsTensor(..)

  , module X
  ) where

import Torch.Core.Tensor.Dynamic.Copy as X

import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import qualified Torch.Class.C.Tensor as Class

import qualified Torch.Core.ByteTensor.Dynamic as B
{-
import qualified Torch.Core.ShortTensor.Dynamic as S
import qualified Torch.Core.IntTensor.Dynamic as I
import qualified Torch.Core.LongTensor.Dynamic as L
import qualified Torch.Core.FloatTensor.Dynamic as F
import qualified Torch.Core.DoubleTensor.Dynamic as D

import qualified Torch.Core.LongStorage as L

type ByteTensor = B.Tensor
-- type CharTensor = C.Tensor
type ShortTensor = S.Tensor
type IntTensor = I.Tensor
type LongTensor = L.Tensor
-- type HalfTensor = H.Tensor
type FloatTensor = F.Tensor
type DoubleTensor = D.Tensor

type LongStorage = L.Storage


instance IsTensor ByteTensor
instance IsTensor ShortTensor
instance IsTensor IntTensor
instance IsTensor LongTensor
instance IsTensor FloatTensor
instance IsTensor DoubleTensor

class C.IsTensor t => IsTensor t where
  clearFlag :: t -> Int8 -> IO ()
  clearFlag = C.clearFlag

  tensordata :: t -> IO [HsReal t]
  tensordata = C.tensordata

  desc :: t -> IO CTHDescBuff
  desc = C.desc

  expand :: t -> t -> [LongStorage] -> IO ()
  -- expand = C.expand

  expandNd :: [t] -> [t] -> Int32 -> IO ()
  -- expandNd = C.expandNd

  get :: t -> Dim (d :: [Nat]) -> IO (HsReal t)
  -- get = C.get

  isContiguous :: t -> IO Bool
  isContiguous = C.isContiguous

  isSameSizeAs :: t -> t -> IO Bool
  isSameSizeAs = C.isSameSizeAs

  isSetTo :: t -> t -> IO Bool
  isSetTo = C.isSetTo

  isSize :: t -> [LongStorage] -> IO Bool
  -- isSize = C.isSize

  nDimension :: t -> IO Int32
  nDimension = C.nDimension

  nElement :: t -> IO Int64
  nElement = C.nElement

  narrow :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  narrow = C.narrow

  new :: IO t
  new = C.new

  newClone :: t -> IO t
  newClone = C.newClone

  newContiguous :: t -> IO t
  newContiguous = C.newContiguous

  newExpand :: t -> [LongStorage] -> IO t
  -- newExpand = C.newExpand

  newNarrow :: t -> Int32 -> Int64 -> Int64 -> IO t
  newNarrow = C.newNarrow

  newSelect :: t -> Int32 -> Int64 -> IO t
  newSelect = C.newSelect

  newSizeOf :: t -> IO [LongStorage]
  -- newSizeOf = C.newSizeOf

  newStrideOf :: t -> IO [LongStorage]
  -- newStrideOf = C.newStrideOf

  newTranspose :: t -> Int32 -> Int32 -> IO t
  newTranspose = C.newTranspose

  newUnfold :: t -> Int32 -> Int64 -> Int64 -> IO t
  newUnfold = C.newUnfold

  newView :: t -> [LongStorage] -> IO t
  -- newView = C.newView

  newWithSize :: [LongStorage] -> [LongStorage] -> IO t
  -- newWithSize = C.newWithSize

  newWithSizeDim :: Dim (d::[Nat]) -> IO t
  -- newWithSizeDim = C.newWithSizeDim

  newWithStorage :: HsStorage t -> Int64 -> [LongStorage] -> [LongStorage] -> IO t
  -- newWithStorage = C.newWithStorage

  newWithStorageDim :: HsStorage t -> Dim (d::[Nat]) -> IO t
  -- newWithStorageDim = C.newWithStorageDim

  newWithTensor :: t -> IO t
  newWithTensor = C.newWithTensor

  resize :: t -> [LongStorage] -> [LongStorage] -> IO ()
  -- resize = C.resize

  resizeDim :: t -> Dim (d::[Nat]) -> IO ()
  -- resizeDim = C.resizeDim

  resizeAs :: t -> t -> IO ()
  resizeAs = C.resizeAs

  resizeNd :: t -> Int32 -> Ptr CLLong -> Ptr CLLong -> IO ()
  resizeNd = C.resizeNd

  retain :: t -> IO ()
  retain = C.retain

  select :: t -> t -> Int32 -> Int64 -> IO ()
  select = C.select

  set :: t -> t -> IO ()
  set = C.set

  setDim :: t -> Dim (d::[Nat]) -> HsReal t -> IO ()
  -- setDim = C.setDim

  setFlag :: t -> Int8 -> IO ()
  setFlag = C.setFlag

  setStorage :: t -> HsStorage t -> Int64 -> [LongStorage] -> [LongStorage] -> IO ()
  -- setStorage = C.setStorage

  setStorageDim :: t -> HsStorage t -> Dim (d::[Nat]) -> IO ()
  -- setStorageDim = C.setStorageDim

  setStorageNd :: t -> HsStorage t -> Int64 -> Int32 -> [CLLong] -> [CLLong] -> IO ()
  -- setStorageNd = C.setStorageNd

  size :: t -> Int32 -> IO Int64
  size = C.size

  sizeDesc :: t -> IO CTHDescBuff
  sizeDesc = C.sizeDesc

  squeeze :: t -> t -> IO ()
  squeeze = C.squeeze

  squeeze1d :: t -> t -> Int32 -> IO ()
  squeeze1d = C.squeeze1d

  storage :: t -> IO (HsStorage t)
  storage = C.storage

  storageOffset :: t -> IO Int64
  storageOffset = C.storageOffset

  stride :: t -> Int32 -> IO Int64
  stride = C.stride

  transpose :: t -> t -> Int32 -> Int32 -> IO ()
  transpose = C.transpose

  unfold :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  unfold = C.unfold

  unsqueeze1d :: t -> t -> Int32 -> IO ()
  unsqueeze1d = C.unsqueeze1d

