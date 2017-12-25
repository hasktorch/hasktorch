{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Generic
  ( flatten
  , randInit
  , constant
  , applyInPlaceFn
  , dimList
  , dimView
  , fillZeros

  , GenericOps(..)
  , GenericMath(..)
  , GenericRandom(..)
  ) where

import Numeric.Dimensions (Dim(..))
import Foreign (Ptr)
import Foreign.C.Types

import Torch.Core.Internal
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.GenericMath
import Torch.Core.Tensor.GenericRandom

import THTypes
import qualified THByteTensor as T
import qualified THDoubleTensor as T
import qualified THFloatTensor as T
import qualified THIntTensor as T
import qualified THLongTensor as T
import qualified THShortTensor as T

-- | flatten a CTHDoubleTensor into a list
flatten :: GenericOps t => Ptr t -> [HaskType t]
flatten tensor =
  case map getDim [0 .. nDimension tensor - 1] of
    []           -> mempty
    [x]          -> get1d tensor <$> range x
    [x, y]       -> get2d tensor <$> range x <*> range y
    [x, y, z]    -> get3d tensor <$> range x <*> range y <*> range z
    [x, y, z, q] -> get4d tensor <$> range x <*> range y <*> range z <*> range q
    _ -> error "TH doesn't support getting tensors higher than 4-dimensions"
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . size tensor

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit
  :: (GenericMath t, GenericRandom t, GenericOps t, Num (HaskType t))
  => Ptr CTHGenerator
  -> Dim (dims :: [k])
  -> CDouble
  -> CDouble
  -> IO (Ptr t)
randInit gen dims lower upper = do
  t <- constant dims 0
  uniform t gen lower upper
  pure t

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
-- fillDouble :: (GenericMath t, GenericOps t) => HaskType t -> Ptr t -> IO ()
-- fillDouble = flip fill . realToFrac

-- | Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
constant :: forall ns t . (GenericMath t, GenericOps t) => Dim (ns::[k]) -> HaskType t -> IO (Ptr t)
constant dims value = do
  newPtr <- go dims
  fill newPtr value
  pure newPtr
  where
    go :: Dim (ns::[k]) -> IO (Ptr t)
    go = onDims fromIntegral
      new
      newWithSize1d
      newWithSize2d
      newWithSize3d
      newWithSize4d

-- |apply a tensor transforming function to a tensor
applyInPlaceFn :: GenericOps t => (Ptr t -> Ptr t -> IO ()) -> Ptr t -> IO (Ptr t)
applyInPlaceFn f t1 = do
  r_ <- new
  f r_ t1
  pure r_

-- |Dimensions of a raw tensor as a list
dimList :: GenericOps t => Ptr t -> [Int]
dimList t = getDim <$> [0 .. nDimension t - 1]
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . size t

-- |Dimensions of a raw tensor as a TensorDim value
dimView :: GenericOps t => Ptr t -> DimView
dimView t =
  case length sz of
    0 -> D0
    1 -> D1 (at 0)
    2 -> D2 (at 0) (at 1)
    3 -> D3 (at 0) (at 1) (at 2)
    4 -> D4 (at 0) (at 1) (at 2) (at 3)
    5 -> D5 (at 0) (at 1) (at 2) (at 3) (at 5)
    _ -> undefined -- TODO - make this safe
  where
    sz :: [Int]
    sz = dimList t

    at :: Int -> Int
    at n = fromIntegral (sz !! n)

-- | Fill a raw Double tensor with 0.0
fillZeros :: (GenericMath t, GenericOps t, Num (HaskType t)) => Ptr t -> IO (Ptr t)
fillZeros t = fill t 0 >> pure t

class GenericOps t where
  type Storage t
  clearFlag :: Ptr t -> CChar -> IO ()
  tensordata :: Ptr t -> IO (Ptr (HaskType t))
  desc :: Ptr t -> CTHDescBuff
  expand :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  expandNd :: Ptr (Ptr t) -> Ptr (Ptr t) -> CInt -> IO ()
  free :: Ptr t -> IO ()
  freeCopyTo :: Ptr t -> Ptr t -> IO ()
  get1d :: Ptr t -> CLLong -> HaskType t
  get2d :: Ptr t -> CLLong -> CLLong -> HaskType t
  get3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t
  get4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t
  isContiguous :: Ptr t -> CInt
  isSameSizeAs :: Ptr t -> Ptr t -> CInt
  isSetTo :: Ptr t -> Ptr t -> CInt
  isSize :: Ptr t -> Ptr CTHLongStorage -> CInt
  nDimension :: Ptr t -> CInt
  nElement :: Ptr t -> CPtrdiff
  narrow :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  new :: IO (Ptr t)
  newClone :: Ptr t -> IO (Ptr t)
  newContiguous :: Ptr t -> IO (Ptr t)
  newExpand :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  newNarrow :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  newSelect :: Ptr t -> CInt -> CLLong -> IO (Ptr t)
  newSizeOf :: Ptr t -> IO (Ptr CTHLongStorage)
  newStrideOf :: Ptr t -> IO (Ptr CTHLongStorage)
  newTranspose :: Ptr t -> CInt -> CInt -> IO (Ptr t)
  newUnfold :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  newView :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithSize1d :: CLLong -> IO (Ptr t)
  newWithSize2d :: CLLong -> CLLong -> IO (Ptr t)
  newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage :: Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithStorage1d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage2d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage3d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage4d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithTensor :: Ptr t -> IO (Ptr t)
  resize :: Ptr t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize1d :: Ptr t -> CLLong -> IO ()
  resize2d :: Ptr t -> CLLong -> CLLong -> IO ()
  resize3d :: Ptr t -> CLLong -> CLLong -> CLLong -> IO ()
  resize4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize5d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resizeAs :: Ptr t -> Ptr t -> IO ()
  resizeNd :: Ptr t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  retain :: Ptr t -> IO ()
  select :: Ptr t -> Ptr t -> CInt -> CLLong -> IO ()
  set :: Ptr t -> Ptr t -> IO ()
  set1d :: Ptr t -> CLLong -> HaskType t -> IO ()
  set2d :: Ptr t -> CLLong -> CLLong -> HaskType t -> IO ()
  set3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ()
  set4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ()
  setFlag :: Ptr t -> CChar -> IO ()
  setStorage :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage1d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO ()
  setStorage2d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage3d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage4d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorageNd :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  size :: Ptr t -> CInt -> CLLong
  sizeDesc :: Ptr t -> CTHDescBuff
  squeeze :: Ptr t -> Ptr t -> IO ()
  squeeze1d :: Ptr t -> Ptr t -> CInt -> IO ()
  storage :: Ptr t -> IO (Ptr (Storage t))
  storageOffset :: Ptr t -> CPtrdiff
  stride :: Ptr t -> CInt -> CLLong
  transpose :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  unfold :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  unsqueeze1d :: Ptr t -> Ptr t -> CInt -> IO ()

instance GenericOps CTHByteTensor where
  type Storage CTHByteTensor = CTHByteStorage
  clearFlag = T.c_THByteTensor_clearFlag
  tensordata = T.c_THByteTensor_data
  desc = T.c_THByteTensor_desc
  expand = T.c_THByteTensor_expand
  expandNd = T.c_THByteTensor_expandNd
  free = T.c_THByteTensor_free
  freeCopyTo = T.c_THByteTensor_freeCopyTo
  get1d = T.c_THByteTensor_get1d
  get2d = T.c_THByteTensor_get2d
  get3d = T.c_THByteTensor_get3d
  get4d = T.c_THByteTensor_get4d
  isContiguous = T.c_THByteTensor_isContiguous
  isSameSizeAs = T.c_THByteTensor_isSameSizeAs
  isSetTo = T.c_THByteTensor_isSetTo
  isSize = T.c_THByteTensor_isSize
  nDimension = T.c_THByteTensor_nDimension
  nElement = T.c_THByteTensor_nElement
  narrow = T.c_THByteTensor_narrow
  new = T.c_THByteTensor_new
  newClone = T.c_THByteTensor_newClone
  newContiguous = T.c_THByteTensor_newContiguous
  newExpand = T.c_THByteTensor_newExpand
  newNarrow = T.c_THByteTensor_newNarrow
  newSelect = T.c_THByteTensor_newSelect
  newSizeOf = T.c_THByteTensor_newSizeOf
  newStrideOf = T.c_THByteTensor_newStrideOf
  newTranspose = T.c_THByteTensor_newTranspose
  newUnfold = T.c_THByteTensor_newUnfold
  newView = T.c_THByteTensor_newView
  newWithSize = T.c_THByteTensor_newWithSize
  newWithSize1d = T.c_THByteTensor_newWithSize1d
  newWithSize2d = T.c_THByteTensor_newWithSize2d
  newWithSize3d = T.c_THByteTensor_newWithSize3d
  newWithSize4d = T.c_THByteTensor_newWithSize4d
  newWithStorage = T.c_THByteTensor_newWithStorage
  newWithStorage1d = T.c_THByteTensor_newWithStorage1d
  newWithStorage2d = T.c_THByteTensor_newWithStorage2d
  newWithStorage3d = T.c_THByteTensor_newWithStorage3d
  newWithStorage4d = T.c_THByteTensor_newWithStorage4d
  newWithTensor = T.c_THByteTensor_newWithTensor
  resize = T.c_THByteTensor_resize
  resize1d = T.c_THByteTensor_resize1d
  resize2d = T.c_THByteTensor_resize2d
  resize3d = T.c_THByteTensor_resize3d
  resize4d = T.c_THByteTensor_resize4d
  resize5d = T.c_THByteTensor_resize5d
  resizeAs = T.c_THByteTensor_resizeAs
  resizeNd = T.c_THByteTensor_resizeNd
  retain = T.c_THByteTensor_retain
  select = T.c_THByteTensor_select
  set = T.c_THByteTensor_set
  set1d = T.c_THByteTensor_set1d
  set2d = T.c_THByteTensor_set2d
  set3d = T.c_THByteTensor_set3d
  set4d = T.c_THByteTensor_set4d
  setFlag = T.c_THByteTensor_setFlag
  setStorage = T.c_THByteTensor_setStorage
  setStorage1d = T.c_THByteTensor_setStorage1d
  setStorage2d = T.c_THByteTensor_setStorage2d
  setStorage3d = T.c_THByteTensor_setStorage3d
  setStorage4d = T.c_THByteTensor_setStorage4d
  setStorageNd = T.c_THByteTensor_setStorageNd
  size = T.c_THByteTensor_size
  sizeDesc = T.c_THByteTensor_sizeDesc
  squeeze = T.c_THByteTensor_squeeze
  squeeze1d = T.c_THByteTensor_squeeze1d
  storage = T.c_THByteTensor_storage
  storageOffset = T.c_THByteTensor_storageOffset
  stride = T.c_THByteTensor_stride
  transpose = T.c_THByteTensor_transpose
  unfold = T.c_THByteTensor_unfold
  unsqueeze1d = T.c_THByteTensor_unsqueeze1d


instance GenericOps CTHDoubleTensor where
  type Storage CTHDoubleTensor = CTHDoubleStorage
  clearFlag = T.c_THDoubleTensor_clearFlag
  tensordata = T.c_THDoubleTensor_data
  desc = T.c_THDoubleTensor_desc
  expand = T.c_THDoubleTensor_expand
  expandNd = T.c_THDoubleTensor_expandNd
  free = T.c_THDoubleTensor_free
  freeCopyTo = T.c_THDoubleTensor_freeCopyTo
  get1d = T.c_THDoubleTensor_get1d
  get2d = T.c_THDoubleTensor_get2d
  get3d = T.c_THDoubleTensor_get3d
  get4d = T.c_THDoubleTensor_get4d
  isContiguous = T.c_THDoubleTensor_isContiguous
  isSameSizeAs = T.c_THDoubleTensor_isSameSizeAs
  isSetTo = T.c_THDoubleTensor_isSetTo
  isSize = T.c_THDoubleTensor_isSize
  nDimension = T.c_THDoubleTensor_nDimension
  nElement = T.c_THDoubleTensor_nElement
  narrow = T.c_THDoubleTensor_narrow
  new = T.c_THDoubleTensor_new
  newClone = T.c_THDoubleTensor_newClone
  newContiguous = T.c_THDoubleTensor_newContiguous
  newExpand = T.c_THDoubleTensor_newExpand
  newNarrow = T.c_THDoubleTensor_newNarrow
  newSelect = T.c_THDoubleTensor_newSelect
  newSizeOf = T.c_THDoubleTensor_newSizeOf
  newStrideOf = T.c_THDoubleTensor_newStrideOf
  newTranspose = T.c_THDoubleTensor_newTranspose
  newUnfold = T.c_THDoubleTensor_newUnfold
  newView = T.c_THDoubleTensor_newView
  newWithSize = T.c_THDoubleTensor_newWithSize
  newWithSize1d = T.c_THDoubleTensor_newWithSize1d
  newWithSize2d = T.c_THDoubleTensor_newWithSize2d
  newWithSize3d = T.c_THDoubleTensor_newWithSize3d
  newWithSize4d = T.c_THDoubleTensor_newWithSize4d
  newWithStorage = T.c_THDoubleTensor_newWithStorage
  newWithStorage1d = T.c_THDoubleTensor_newWithStorage1d
  newWithStorage2d = T.c_THDoubleTensor_newWithStorage2d
  newWithStorage3d = T.c_THDoubleTensor_newWithStorage3d
  newWithStorage4d = T.c_THDoubleTensor_newWithStorage4d
  newWithTensor = T.c_THDoubleTensor_newWithTensor
  resize = T.c_THDoubleTensor_resize
  resize1d = T.c_THDoubleTensor_resize1d
  resize2d = T.c_THDoubleTensor_resize2d
  resize3d = T.c_THDoubleTensor_resize3d
  resize4d = T.c_THDoubleTensor_resize4d
  resize5d = T.c_THDoubleTensor_resize5d
  resizeAs = T.c_THDoubleTensor_resizeAs
  resizeNd = T.c_THDoubleTensor_resizeNd
  retain = T.c_THDoubleTensor_retain
  select = T.c_THDoubleTensor_select
  set = T.c_THDoubleTensor_set
  set1d = T.c_THDoubleTensor_set1d
  set2d = T.c_THDoubleTensor_set2d
  set3d = T.c_THDoubleTensor_set3d
  set4d = T.c_THDoubleTensor_set4d
  setFlag = T.c_THDoubleTensor_setFlag
  setStorage = T.c_THDoubleTensor_setStorage
  setStorage1d = T.c_THDoubleTensor_setStorage1d
  setStorage2d = T.c_THDoubleTensor_setStorage2d
  setStorage3d = T.c_THDoubleTensor_setStorage3d
  setStorage4d = T.c_THDoubleTensor_setStorage4d
  setStorageNd = T.c_THDoubleTensor_setStorageNd
  size = T.c_THDoubleTensor_size
  sizeDesc = T.c_THDoubleTensor_sizeDesc
  squeeze = T.c_THDoubleTensor_squeeze
  squeeze1d = T.c_THDoubleTensor_squeeze1d
  storage = T.c_THDoubleTensor_storage
  storageOffset = T.c_THDoubleTensor_storageOffset
  stride = T.c_THDoubleTensor_stride
  transpose = T.c_THDoubleTensor_transpose
  unfold = T.c_THDoubleTensor_unfold
  unsqueeze1d = T.c_THDoubleTensor_unsqueeze1d

instance GenericOps CTHFloatTensor where
  type Storage CTHFloatTensor = CTHFloatStorage
  clearFlag = T.c_THFloatTensor_clearFlag
  tensordata = T.c_THFloatTensor_data
  desc = T.c_THFloatTensor_desc
  expand = T.c_THFloatTensor_expand
  expandNd = T.c_THFloatTensor_expandNd
  free = T.c_THFloatTensor_free
  freeCopyTo = T.c_THFloatTensor_freeCopyTo
  get1d = T.c_THFloatTensor_get1d
  get2d = T.c_THFloatTensor_get2d
  get3d = T.c_THFloatTensor_get3d
  get4d = T.c_THFloatTensor_get4d
  isContiguous = T.c_THFloatTensor_isContiguous
  isSameSizeAs = T.c_THFloatTensor_isSameSizeAs
  isSetTo = T.c_THFloatTensor_isSetTo
  isSize = T.c_THFloatTensor_isSize
  nDimension = T.c_THFloatTensor_nDimension
  nElement = T.c_THFloatTensor_nElement
  narrow = T.c_THFloatTensor_narrow
  new = T.c_THFloatTensor_new
  newClone = T.c_THFloatTensor_newClone
  newContiguous = T.c_THFloatTensor_newContiguous
  newExpand = T.c_THFloatTensor_newExpand
  newNarrow = T.c_THFloatTensor_newNarrow
  newSelect = T.c_THFloatTensor_newSelect
  newSizeOf = T.c_THFloatTensor_newSizeOf
  newStrideOf = T.c_THFloatTensor_newStrideOf
  newTranspose = T.c_THFloatTensor_newTranspose
  newUnfold = T.c_THFloatTensor_newUnfold
  newView = T.c_THFloatTensor_newView
  newWithSize = T.c_THFloatTensor_newWithSize
  newWithSize1d = T.c_THFloatTensor_newWithSize1d
  newWithSize2d = T.c_THFloatTensor_newWithSize2d
  newWithSize3d = T.c_THFloatTensor_newWithSize3d
  newWithSize4d = T.c_THFloatTensor_newWithSize4d
  newWithStorage = T.c_THFloatTensor_newWithStorage
  newWithStorage1d = T.c_THFloatTensor_newWithStorage1d
  newWithStorage2d = T.c_THFloatTensor_newWithStorage2d
  newWithStorage3d = T.c_THFloatTensor_newWithStorage3d
  newWithStorage4d = T.c_THFloatTensor_newWithStorage4d
  newWithTensor = T.c_THFloatTensor_newWithTensor
  resize = T.c_THFloatTensor_resize
  resize1d = T.c_THFloatTensor_resize1d
  resize2d = T.c_THFloatTensor_resize2d
  resize3d = T.c_THFloatTensor_resize3d
  resize4d = T.c_THFloatTensor_resize4d
  resize5d = T.c_THFloatTensor_resize5d
  resizeAs = T.c_THFloatTensor_resizeAs
  resizeNd = T.c_THFloatTensor_resizeNd
  retain = T.c_THFloatTensor_retain
  select = T.c_THFloatTensor_select
  set = T.c_THFloatTensor_set
  set1d = T.c_THFloatTensor_set1d
  set2d = T.c_THFloatTensor_set2d
  set3d = T.c_THFloatTensor_set3d
  set4d = T.c_THFloatTensor_set4d
  setFlag = T.c_THFloatTensor_setFlag
  setStorage = T.c_THFloatTensor_setStorage
  setStorage1d = T.c_THFloatTensor_setStorage1d
  setStorage2d = T.c_THFloatTensor_setStorage2d
  setStorage3d = T.c_THFloatTensor_setStorage3d
  setStorage4d = T.c_THFloatTensor_setStorage4d
  setStorageNd = T.c_THFloatTensor_setStorageNd
  size = T.c_THFloatTensor_size
  sizeDesc = T.c_THFloatTensor_sizeDesc
  squeeze = T.c_THFloatTensor_squeeze
  squeeze1d = T.c_THFloatTensor_squeeze1d
  storage = T.c_THFloatTensor_storage
  storageOffset = T.c_THFloatTensor_storageOffset
  stride = T.c_THFloatTensor_stride
  transpose = T.c_THFloatTensor_transpose
  unfold = T.c_THFloatTensor_unfold
  unsqueeze1d = T.c_THFloatTensor_unsqueeze1d

instance GenericOps CTHIntTensor where
  type Storage CTHIntTensor = CTHIntStorage
  clearFlag = T.c_THIntTensor_clearFlag
  tensordata = T.c_THIntTensor_data
  desc = T.c_THIntTensor_desc
  expand = T.c_THIntTensor_expand
  expandNd = T.c_THIntTensor_expandNd
  free = T.c_THIntTensor_free
  freeCopyTo = T.c_THIntTensor_freeCopyTo
  get1d = T.c_THIntTensor_get1d
  get2d = T.c_THIntTensor_get2d
  get3d = T.c_THIntTensor_get3d
  get4d = T.c_THIntTensor_get4d
  isContiguous = T.c_THIntTensor_isContiguous
  isSameSizeAs = T.c_THIntTensor_isSameSizeAs
  isSetTo = T.c_THIntTensor_isSetTo
  isSize = T.c_THIntTensor_isSize
  nDimension = T.c_THIntTensor_nDimension
  nElement = T.c_THIntTensor_nElement
  narrow = T.c_THIntTensor_narrow
  new = T.c_THIntTensor_new
  newClone = T.c_THIntTensor_newClone
  newContiguous = T.c_THIntTensor_newContiguous
  newExpand = T.c_THIntTensor_newExpand
  newNarrow = T.c_THIntTensor_newNarrow
  newSelect = T.c_THIntTensor_newSelect
  newSizeOf = T.c_THIntTensor_newSizeOf
  newStrideOf = T.c_THIntTensor_newStrideOf
  newTranspose = T.c_THIntTensor_newTranspose
  newUnfold = T.c_THIntTensor_newUnfold
  newView = T.c_THIntTensor_newView
  newWithSize = T.c_THIntTensor_newWithSize
  newWithSize1d = T.c_THIntTensor_newWithSize1d
  newWithSize2d = T.c_THIntTensor_newWithSize2d
  newWithSize3d = T.c_THIntTensor_newWithSize3d
  newWithSize4d = T.c_THIntTensor_newWithSize4d
  newWithStorage = T.c_THIntTensor_newWithStorage
  newWithStorage1d = T.c_THIntTensor_newWithStorage1d
  newWithStorage2d = T.c_THIntTensor_newWithStorage2d
  newWithStorage3d = T.c_THIntTensor_newWithStorage3d
  newWithStorage4d = T.c_THIntTensor_newWithStorage4d
  newWithTensor = T.c_THIntTensor_newWithTensor
  resize = T.c_THIntTensor_resize
  resize1d = T.c_THIntTensor_resize1d
  resize2d = T.c_THIntTensor_resize2d
  resize3d = T.c_THIntTensor_resize3d
  resize4d = T.c_THIntTensor_resize4d
  resize5d = T.c_THIntTensor_resize5d
  resizeAs = T.c_THIntTensor_resizeAs
  resizeNd = T.c_THIntTensor_resizeNd
  retain = T.c_THIntTensor_retain
  select = T.c_THIntTensor_select
  set = T.c_THIntTensor_set
  set1d = T.c_THIntTensor_set1d
  set2d = T.c_THIntTensor_set2d
  set3d = T.c_THIntTensor_set3d
  set4d = T.c_THIntTensor_set4d
  setFlag = T.c_THIntTensor_setFlag
  setStorage = T.c_THIntTensor_setStorage
  setStorage1d = T.c_THIntTensor_setStorage1d
  setStorage2d = T.c_THIntTensor_setStorage2d
  setStorage3d = T.c_THIntTensor_setStorage3d
  setStorage4d = T.c_THIntTensor_setStorage4d
  setStorageNd = T.c_THIntTensor_setStorageNd
  size = T.c_THIntTensor_size
  sizeDesc = T.c_THIntTensor_sizeDesc
  squeeze = T.c_THIntTensor_squeeze
  squeeze1d = T.c_THIntTensor_squeeze1d
  storage = T.c_THIntTensor_storage
  storageOffset = T.c_THIntTensor_storageOffset
  stride = T.c_THIntTensor_stride
  transpose = T.c_THIntTensor_transpose
  unfold = T.c_THIntTensor_unfold
  unsqueeze1d = T.c_THIntTensor_unsqueeze1d

instance GenericOps CTHLongTensor where
  type Storage CTHLongTensor = CTHLongStorage
  clearFlag = T.c_THLongTensor_clearFlag
  tensordata = T.c_THLongTensor_data
  desc = T.c_THLongTensor_desc
  expand = T.c_THLongTensor_expand
  expandNd = T.c_THLongTensor_expandNd
  free = T.c_THLongTensor_free
  freeCopyTo = T.c_THLongTensor_freeCopyTo
  get1d = T.c_THLongTensor_get1d
  get2d = T.c_THLongTensor_get2d
  get3d = T.c_THLongTensor_get3d
  get4d = T.c_THLongTensor_get4d
  isContiguous = T.c_THLongTensor_isContiguous
  isSameSizeAs = T.c_THLongTensor_isSameSizeAs
  isSetTo = T.c_THLongTensor_isSetTo
  isSize = T.c_THLongTensor_isSize
  nDimension = T.c_THLongTensor_nDimension
  nElement = T.c_THLongTensor_nElement
  narrow = T.c_THLongTensor_narrow
  new = T.c_THLongTensor_new
  newClone = T.c_THLongTensor_newClone
  newContiguous = T.c_THLongTensor_newContiguous
  newExpand = T.c_THLongTensor_newExpand
  newNarrow = T.c_THLongTensor_newNarrow
  newSelect = T.c_THLongTensor_newSelect
  newSizeOf = T.c_THLongTensor_newSizeOf
  newStrideOf = T.c_THLongTensor_newStrideOf
  newTranspose = T.c_THLongTensor_newTranspose
  newUnfold = T.c_THLongTensor_newUnfold
  newView = T.c_THLongTensor_newView
  newWithSize = T.c_THLongTensor_newWithSize
  newWithSize1d = T.c_THLongTensor_newWithSize1d
  newWithSize2d = T.c_THLongTensor_newWithSize2d
  newWithSize3d = T.c_THLongTensor_newWithSize3d
  newWithSize4d = T.c_THLongTensor_newWithSize4d
  newWithStorage = T.c_THLongTensor_newWithStorage
  newWithStorage1d = T.c_THLongTensor_newWithStorage1d
  newWithStorage2d = T.c_THLongTensor_newWithStorage2d
  newWithStorage3d = T.c_THLongTensor_newWithStorage3d
  newWithStorage4d = T.c_THLongTensor_newWithStorage4d
  newWithTensor = T.c_THLongTensor_newWithTensor
  resize = T.c_THLongTensor_resize
  resize1d = T.c_THLongTensor_resize1d
  resize2d = T.c_THLongTensor_resize2d
  resize3d = T.c_THLongTensor_resize3d
  resize4d = T.c_THLongTensor_resize4d
  resize5d = T.c_THLongTensor_resize5d
  resizeAs = T.c_THLongTensor_resizeAs
  resizeNd = T.c_THLongTensor_resizeNd
  retain = T.c_THLongTensor_retain
  select = T.c_THLongTensor_select
  set = T.c_THLongTensor_set
  set1d = T.c_THLongTensor_set1d
  set2d = T.c_THLongTensor_set2d
  set3d = T.c_THLongTensor_set3d
  set4d = T.c_THLongTensor_set4d
  setFlag = T.c_THLongTensor_setFlag
  setStorage = T.c_THLongTensor_setStorage
  setStorage1d = T.c_THLongTensor_setStorage1d
  setStorage2d = T.c_THLongTensor_setStorage2d
  setStorage3d = T.c_THLongTensor_setStorage3d
  setStorage4d = T.c_THLongTensor_setStorage4d
  setStorageNd = T.c_THLongTensor_setStorageNd
  size = T.c_THLongTensor_size
  sizeDesc = T.c_THLongTensor_sizeDesc
  squeeze = T.c_THLongTensor_squeeze
  squeeze1d = T.c_THLongTensor_squeeze1d
  storage = T.c_THLongTensor_storage
  storageOffset = T.c_THLongTensor_storageOffset
  stride = T.c_THLongTensor_stride
  transpose = T.c_THLongTensor_transpose
  unfold = T.c_THLongTensor_unfold
  unsqueeze1d = T.c_THLongTensor_unsqueeze1d


instance GenericOps CTHShortTensor where
  type Storage CTHShortTensor = CTHShortStorage
  clearFlag = T.c_THShortTensor_clearFlag
  tensordata = T.c_THShortTensor_data
  desc = T.c_THShortTensor_desc
  expand = T.c_THShortTensor_expand
  expandNd = T.c_THShortTensor_expandNd
  free = T.c_THShortTensor_free
  freeCopyTo = T.c_THShortTensor_freeCopyTo
  get1d = T.c_THShortTensor_get1d
  get2d = T.c_THShortTensor_get2d
  get3d = T.c_THShortTensor_get3d
  get4d = T.c_THShortTensor_get4d
  isContiguous = T.c_THShortTensor_isContiguous
  isSameSizeAs = T.c_THShortTensor_isSameSizeAs
  isSetTo = T.c_THShortTensor_isSetTo
  isSize = T.c_THShortTensor_isSize
  nDimension = T.c_THShortTensor_nDimension
  nElement = T.c_THShortTensor_nElement
  narrow = T.c_THShortTensor_narrow
  new = T.c_THShortTensor_new
  newClone = T.c_THShortTensor_newClone
  newContiguous = T.c_THShortTensor_newContiguous
  newExpand = T.c_THShortTensor_newExpand
  newNarrow = T.c_THShortTensor_newNarrow
  newSelect = T.c_THShortTensor_newSelect
  newSizeOf = T.c_THShortTensor_newSizeOf
  newStrideOf = T.c_THShortTensor_newStrideOf
  newTranspose = T.c_THShortTensor_newTranspose
  newUnfold = T.c_THShortTensor_newUnfold
  newView = T.c_THShortTensor_newView
  newWithSize = T.c_THShortTensor_newWithSize
  newWithSize1d = T.c_THShortTensor_newWithSize1d
  newWithSize2d = T.c_THShortTensor_newWithSize2d
  newWithSize3d = T.c_THShortTensor_newWithSize3d
  newWithSize4d = T.c_THShortTensor_newWithSize4d
  newWithStorage = T.c_THShortTensor_newWithStorage
  newWithStorage1d = T.c_THShortTensor_newWithStorage1d
  newWithStorage2d = T.c_THShortTensor_newWithStorage2d
  newWithStorage3d = T.c_THShortTensor_newWithStorage3d
  newWithStorage4d = T.c_THShortTensor_newWithStorage4d
  newWithTensor = T.c_THShortTensor_newWithTensor
  resize = T.c_THShortTensor_resize
  resize1d = T.c_THShortTensor_resize1d
  resize2d = T.c_THShortTensor_resize2d
  resize3d = T.c_THShortTensor_resize3d
  resize4d = T.c_THShortTensor_resize4d
  resize5d = T.c_THShortTensor_resize5d
  resizeAs = T.c_THShortTensor_resizeAs
  resizeNd = T.c_THShortTensor_resizeNd
  retain = T.c_THShortTensor_retain
  select = T.c_THShortTensor_select
  set = T.c_THShortTensor_set
  set1d = T.c_THShortTensor_set1d
  set2d = T.c_THShortTensor_set2d
  set3d = T.c_THShortTensor_set3d
  set4d = T.c_THShortTensor_set4d
  setFlag = T.c_THShortTensor_setFlag
  setStorage = T.c_THShortTensor_setStorage
  setStorage1d = T.c_THShortTensor_setStorage1d
  setStorage2d = T.c_THShortTensor_setStorage2d
  setStorage3d = T.c_THShortTensor_setStorage3d
  setStorage4d = T.c_THShortTensor_setStorage4d
  setStorageNd = T.c_THShortTensor_setStorageNd
  size = T.c_THShortTensor_size
  sizeDesc = T.c_THShortTensor_sizeDesc
  squeeze = T.c_THShortTensor_squeeze
  squeeze1d = T.c_THShortTensor_squeeze1d
  storage = T.c_THShortTensor_storage
  storageOffset = T.c_THShortTensor_storageOffset
  stride = T.c_THShortTensor_stride
  transpose = T.c_THShortTensor_transpose
  unfold = T.c_THShortTensor_unfold
  unsqueeze1d = T.c_THShortTensor_unsqueeze1d


