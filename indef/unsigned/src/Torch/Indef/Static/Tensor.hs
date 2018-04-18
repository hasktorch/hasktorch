{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Static.Tensor () where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor as Dynamic
import qualified Torch.Class.Tensor.Static as Class
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Types as Sig
import Data.Coerce

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()

-- Definitely don't export this. Make sure these gory details never see the day of light.
sudo :: Tensor d -> Tensor d'
sudo t = Sig.asStatic ((Sig.asDynamic t) :: Dynamic)

instance Class.IsTensor Tensor where
  isSameSizeAs :: forall t d d' . (Dimensions d', Dimensions d) => t d -> t d' -> Bool
  isSameSizeAs _ _ = dimVals (dim :: Dim d) == dimVals (dim :: Dim d')

  fromList1d :: [HsReal] -> IO (Tensor '[n])
  fromList1d l = asStatic <$> (Dynamic.fromList1d l)

  newExpand t = fmap asStatic . Dynamic.newExpand (asDynamic t)
  _expand r t = Dynamic._expand (asDynamic r) (asDynamic t)
  _expandNd rs os = Dynamic._expandNd (fmap asDynamic rs) (fmap asDynamic os)

  _resize t a b = Dynamic._resize (asDynamic t) a b >> pure (sudo t)
  _resize1d t a = Dynamic._resize1d (asDynamic t) a >> pure (sudo t)
  _resize2d t a b = Dynamic._resize2d (asDynamic t) a b >> pure (sudo t)
  _resize3d t a b c = Dynamic._resize3d (asDynamic t) a b c >> pure (sudo t)
  _resize4d t a b c d = Dynamic._resize4d (asDynamic t) a b c d >> pure (sudo t)
  _resize5d t a b c d e = Dynamic._resize5d (asDynamic t) a b c d e >> pure (sudo t)
  _resizeAs src tar = Dynamic._resizeAs (asDynamic src) (asDynamic tar) >> pure (sudo src)
  _resizeNd src a b c = Dynamic._resizeNd (asDynamic src) a b c >> pure (sudo src)
  retain t = Dynamic.retain (asDynamic t)
  _clearFlag t = Dynamic._clearFlag (asDynamic t)
  tensordata t = Dynamic.tensordata (asDynamic t)
  _free t = Dynamic._free (asDynamic t)
  _freeCopyTo t0 t1 = Dynamic._freeCopyTo (asDynamic t0) (asDynamic t1)
  get1d t = Dynamic.get1d (asDynamic t)
  get2d t = Dynamic.get2d (asDynamic t)
  get3d t = Dynamic.get3d (asDynamic t)
  get4d t = Dynamic.get4d (asDynamic t)
  isContiguous t = Dynamic.isContiguous (asDynamic t)

  isSetTo t0 t1 = Dynamic.isSetTo (asDynamic t0) (asDynamic t1)
  isSize t = Dynamic.isSize (asDynamic t)
  nDimension t = Dynamic.nDimension (asDynamic t)
  nElement t = Dynamic.nElement (asDynamic t)
  _narrow t0 t1 = Dynamic._narrow (asDynamic t0) (asDynamic t1)
  empty = asStatic <$> Dynamic.empty
  newClone t = asStatic <$> Dynamic.newClone (asDynamic t)
  newContiguous t = asStatic <$> Dynamic.newContiguous (asDynamic t)
  newNarrow t a b c = asStatic <$> Dynamic.newNarrow (asDynamic t) a b c
  newSelect t a b = asStatic <$> Dynamic.newSelect (asDynamic t) a b
  newSizeOf t = Dynamic.newSizeOf (asDynamic t)
  newStrideOf t = Dynamic.newStrideOf (asDynamic t)
  newTranspose t a b = asStatic <$> Dynamic.newTranspose (asDynamic t) a b
  newUnfold t a b c = asStatic <$> Dynamic.newUnfold (asDynamic t) a b c
  newView t a = asStatic <$> Dynamic.newView (asDynamic t) a
  newWithSize a0 a1 = asStatic <$> Dynamic.newWithSize a0 a1
  newWithSize1d a0 = asStatic <$> Dynamic.newWithSize1d a0
  newWithSize2d a0 a1 = asStatic <$> Dynamic.newWithSize2d a0 a1
  newWithSize3d a0 a1 a2 = asStatic <$> Dynamic.newWithSize3d a0 a1 a2
  newWithSize4d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithSize4d a0 a1 a2 a3
  newWithStorage a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage a0 a1 a2 a3
  newWithStorage1d a0 a1 a2 = asStatic <$> Dynamic.newWithStorage1d a0 a1 a2
  newWithStorage2d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage2d a0 a1 a2 a3
  newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Dynamic.newWithStorage3d a0 a1 a2 a3 a4
  newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Dynamic.newWithStorage4d a0 a1 a2 a3 a4 a5
  newWithTensor t = asStatic <$> Dynamic.newWithTensor (asDynamic t)
  _select t0 t1 = Dynamic._select (asDynamic t0) (asDynamic t1)
  _set t0 t1 = Dynamic._set (asDynamic t0) (asDynamic t1)
  _set1d t = Dynamic._set1d (asDynamic t)
  _set2d t = Dynamic._set2d (asDynamic t)
  _set3d t = Dynamic._set3d (asDynamic t)
  _set4d t = Dynamic._set4d (asDynamic t)
  _setFlag t = Dynamic._setFlag (asDynamic t)
  _setStorage t = Dynamic._setStorage (asDynamic t)
  _setStorage1d t = Dynamic._setStorage1d (asDynamic t)
  _setStorage2d t = Dynamic._setStorage2d (asDynamic t)
  _setStorage3d t = Dynamic._setStorage3d (asDynamic t)
  _setStorage4d t = Dynamic._setStorage4d (asDynamic t)
  _setStorageNd t = Dynamic._setStorageNd (asDynamic t)
  size t = Dynamic.size (asDynamic t)
  sizeDesc t = Dynamic.sizeDesc (asDynamic t)
  _squeeze t0 t1 = Dynamic._squeeze (asDynamic t0) (asDynamic t1)
  _squeeze1d t0 t1 = Dynamic._squeeze1d (asDynamic t0) (asDynamic t1)
  storage t = Dynamic.storage (asDynamic t)
  storageOffset t = Dynamic.storageOffset (asDynamic t)
  stride t = Dynamic.stride (asDynamic t)
  _transpose t0 t1 = Dynamic._transpose (asDynamic t0) (asDynamic t1)
  _unfold t0 t1 = Dynamic._unfold (asDynamic t0) (asDynamic t1)
  _unsqueeze1d t0 t1 = Dynamic._unsqueeze1d (asDynamic t0) (asDynamic t1)



