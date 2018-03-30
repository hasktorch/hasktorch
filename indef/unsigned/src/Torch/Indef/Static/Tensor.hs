{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Static.Tensor where

import Torch.Dimensions (DimVal)
import qualified Torch.Class.Tensor as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()

-- TODO: replace some of this with static-aware versions
instance Class.Tensor (Tensor d) where
  clearFlag_ t = Class.clearFlag_ (asDynamic t)
  tensordata t = Class.tensordata (asDynamic t)
  free_ t = Class.free_ (asDynamic t)
  freeCopyTo_ t0 t1 = Class.freeCopyTo_ (asDynamic t0) (asDynamic t1)
  get1d t = Class.get1d (asDynamic t)
  get2d t = Class.get2d (asDynamic t)
  get3d t = Class.get3d (asDynamic t)
  get4d t = Class.get4d (asDynamic t)
  isContiguous t = Class.isContiguous (asDynamic t)
  isSameSizeAs t0 t1 = Class.isSameSizeAs (asDynamic t0) (asDynamic t1)
  isSetTo t0 t1 = Class.isSetTo (asDynamic t0) (asDynamic t1)
  isSize t = Class.isSize (asDynamic t)
  nDimension t = Class.nDimension (asDynamic t)
  nElement t = Class.nElement (asDynamic t)
  narrow_ t0 t1 = Class.narrow_ (asDynamic t0) (asDynamic t1)
  empty = asStatic <$> Class.empty
  newClone t = asStatic <$> Class.newClone (asDynamic t)
  newContiguous t = asStatic <$> Class.newContiguous (asDynamic t)
  newNarrow t a b c = asStatic <$> Class.newNarrow (asDynamic t) a b c
  newSelect t a b = asStatic <$> Class.newSelect (asDynamic t) a b
  newSizeOf t = Class.newSizeOf (asDynamic t)
  newStrideOf t = Class.newStrideOf (asDynamic t)
  newTranspose t a b = asStatic <$> Class.newTranspose (asDynamic t) a b
  newUnfold t a b c = asStatic <$> Class.newUnfold (asDynamic t) a b c
  newView t a = asStatic <$> Class.newView (asDynamic t) a
  newWithSize a0 a1 = asStatic <$> Class.newWithSize a0 a1
  newWithSize1d a0 = asStatic <$> Class.newWithSize1d a0
  newWithSize2d a0 a1 = asStatic <$> Class.newWithSize2d a0 a1
  newWithSize3d a0 a1 a2 = asStatic <$> Class.newWithSize3d a0 a1 a2
  newWithSize4d a0 a1 a2 a3 = asStatic <$> Class.newWithSize4d a0 a1 a2 a3
  newWithStorage a0 a1 a2 a3 = asStatic <$> Class.newWithStorage a0 a1 a2 a3
  newWithStorage1d a0 a1 a2 = asStatic <$> Class.newWithStorage1d a0 a1 a2
  newWithStorage2d a0 a1 a2 a3 = asStatic <$> Class.newWithStorage2d a0 a1 a2 a3
  newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Class.newWithStorage3d a0 a1 a2 a3 a4
  newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Class.newWithStorage4d a0 a1 a2 a3 a4 a5
  newWithTensor t = asStatic <$> Class.newWithTensor (asDynamic t)
  resize_ t = Class.resize_ (asDynamic t)
  resize1d_ t = Class.resize1d_ (asDynamic t)
  resize2d_ t = Class.resize2d_ (asDynamic t)
  resize3d_ t = Class.resize3d_ (asDynamic t)
  resize4d_ t = Class.resize4d_ (asDynamic t)
  resize5d_ t = Class.resize5d_ (asDynamic t)
  resizeAs_ t0 t1 = Class.resizeAs_ (asDynamic t0) (asDynamic t1)
  resizeNd_ t = Class.resizeNd_ (asDynamic t)
  retain t = Class.retain (asDynamic t)
  select_ t0 t1 = Class.select_ (asDynamic t0) (asDynamic t1)
  set_ t0 t1 = Class.set_ (asDynamic t0) (asDynamic t1)
  set1d_ t = Class.set1d_ (asDynamic t)
  set2d_ t = Class.set2d_ (asDynamic t)
  set3d_ t = Class.set3d_ (asDynamic t)
  set4d_ t = Class.set4d_ (asDynamic t)
  setFlag_ t = Class.setFlag_ (asDynamic t)
  setStorage_ t = Class.setStorage_ (asDynamic t)
  setStorage1d_ t = Class.setStorage1d_ (asDynamic t)
  setStorage2d_ t = Class.setStorage2d_ (asDynamic t)
  setStorage3d_ t = Class.setStorage3d_ (asDynamic t)
  setStorage4d_ t = Class.setStorage4d_ (asDynamic t)
  setStorageNd_ t = Class.setStorageNd_ (asDynamic t)
  size t = Class.size (asDynamic t)
  sizeDesc t = Class.sizeDesc (asDynamic t)
  squeeze_ t0 t1 = Class.squeeze_ (asDynamic t0) (asDynamic t1)
  squeeze1d_ t0 t1 = Class.squeeze1d_ (asDynamic t0) (asDynamic t1)
  storage t = Class.storage (asDynamic t)
  storageOffset t = Class.storageOffset (asDynamic t)
  stride t = Class.stride (asDynamic t)
  transpose_ t0 t1 = Class.transpose_ (asDynamic t0) (asDynamic t1)
  unfold_ t0 t1 = Class.unfold_ (asDynamic t0) (asDynamic t1)
  unsqueeze1d_ t0 t1 = Class.unsqueeze1d_ (asDynamic t0) (asDynamic t1)

