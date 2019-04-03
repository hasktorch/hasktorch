-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.CompareT
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Compare two tensors
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.CompareT
  ( ltTensor, ltTensorT, ltTensorT_
  , leTensor, leTensorT, leTensorT_
  , gtTensor, gtTensorT, gtTensorT_
  , geTensor, geTensorT, geTensorT_
  , neTensor, neTensorT, neTensorT_
  , eqTensor, eqTensorT, eqTensorT_
  ) where

import Foreign hiding (with, new)
import Foreign.Ptr
import System.IO.Unsafe
import Numeric.Dimensions
import qualified Torch.Sig.Tensor.Math.CompareT as Sig

import Torch.Indef.Mask
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

_ltTensorT, _leTensorT, _gtTensorT, _geTensorT, _neTensorT, _eqTensorT
  :: Dynamic -> Dynamic -> Dynamic -> IO ()
_ltTensorT = compareTensorTOp Sig.c_ltTensorT
_leTensorT = compareTensorTOp Sig.c_leTensorT
_gtTensorT = compareTensorTOp Sig.c_gtTensorT
_geTensorT = compareTensorTOp Sig.c_geTensorT
_neTensorT = compareTensorTOp Sig.c_neTensorT
_eqTensorT = compareTensorTOp Sig.c_eqTensorT

compareTensorTOp
  :: (Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> Dynamic -> Dynamic -> IO ()
compareTensorTOp fn a b c = withLift $ fn
  <$> managedState
  <*> managedTensor a
  <*> managedTensor b
  <*> managedTensor c


compareTensorOp
  :: (Ptr CState -> Ptr CByteTensor -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> Dynamic -> MaskDynamic
compareTensorOp op t0 t1 = unsafeDupablePerformIO $ do
  let
    sd = getSomeDims t0
    bt = newMaskDyn' sd
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    withMask bt $ \bt' -> op s' bt' t0' t1'
  pure bt
{-# NOINLINE compareTensorOp #-}

-- | Return a byte tensor which contains boolean values indicating the relation between two tensors.
ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: Dynamic -> Dynamic -> MaskDynamic
ltTensor = compareTensorOp Sig.c_ltTensor
leTensor = compareTensorOp Sig.c_leTensor
gtTensor = compareTensorOp Sig.c_gtTensor
geTensor = compareTensorOp Sig.c_geTensor
neTensor = compareTensorOp Sig.c_neTensor
eqTensor = compareTensorOp Sig.c_eqTensor

-- | return a tensor which contains numeric values indicating the relation between two tensors.
-- 0 stands for false, 1 stands for true.
ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dynamic  -- ^ source tensor.
  -> Dynamic  -- ^ tensor to compare with.
  -> Dynamic  -- ^ new return tensor.
ltTensorT  a b = unsafeDupablePerformIO $ let r = empty in _ltTensorT r a b >> pure r
leTensorT  a b = unsafeDupablePerformIO $ let r = empty in _leTensorT r a b >> pure r
gtTensorT  a b = unsafeDupablePerformIO $ let r = empty in _gtTensorT r a b >> pure r
geTensorT  a b = unsafeDupablePerformIO $ let r = empty in _geTensorT r a b >> pure r
neTensorT  a b = unsafeDupablePerformIO $ let r = empty in _neTensorT r a b >> pure r
eqTensorT  a b = unsafeDupablePerformIO $ let r = empty in _eqTensorT r a b >> pure r
{-# NOINLINE ltTensorT #-}
{-# NOINLINE leTensorT #-}
{-# NOINLINE gtTensorT #-}
{-# NOINLINE geTensorT #-}
{-# NOINLINE neTensorT #-}
{-# NOINLINE eqTensorT #-}

-- | mutate a tensor in-place with its numeric relation to the second tensor of the same size,
-- where 0 stands for false and 1 stands for true.
ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: Dynamic  -- ^ source tensor to mutate inplace.
  -> Dynamic  -- ^ tensor to compare with.
  -> IO ()
ltTensorT_ a b = _ltTensorT a a b
leTensorT_ a b = _leTensorT a a b
gtTensorT_ a b = _gtTensorT a a b
geTensorT_ a b = _geTensorT a a b
neTensorT_ a b = _neTensorT a a b
eqTensorT_ a b = _eqTensorT a a b

