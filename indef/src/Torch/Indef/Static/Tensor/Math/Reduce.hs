-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Reduce
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.Math.Reduce
  ( minall
  , maxall
  , medianall
  , sumall
  , prodall

  , Torch.Indef.Static.Tensor.Math.Reduce.min
  , Torch.Indef.Static.Tensor.Math.Reduce.max
  , median

  -- , minIndex1d    -- , min1d
  -- , maxIndex1d    -- , max1d
  -- , medianIndex1d -- , median1d

  , Torch.Indef.Static.Tensor.Math.Reduce.sum, rowsum, colsum
  , _prod
  ) where

import Numeric.Dimensions
import Data.Coerce
import System.IO.Unsafe
import Data.Singletons.Prelude.List hiding (All, type (++), Length)
import Data.Singletons.Prelude.Ord
import GHC.TypeLits


import Data.Maybe (fromJust)
import Torch.Indef.Index
import Torch.Indef.Static.Tensor
import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Reduce as Dynamic


-- | Static call to 'Dynamic.minall'
minall :: Tensor d -> HsReal
minall t = Dynamic.minall (asDynamic t)

-- | Static call to 'Dynamic.maxall'
maxall :: Tensor d -> HsReal
maxall t = Dynamic.maxall (asDynamic t)

-- | Static call to 'Dynamic.medianall'
medianall :: Tensor d -> HsReal
medianall t = Dynamic.medianall (asDynamic t)

-- | Static call to 'Dynamic.sumall'
sumall :: Tensor d -> HsAccReal
sumall t = Dynamic.sumall (asDynamic t)

-- | Static call to 'Dynamic.prodall'
prodall :: Tensor d -> HsAccReal
prodall t = Dynamic.prodall (asDynamic t)

-- | Static call to 'Dynamic.max'
max
  :: forall d n ix rs ls
  .  All Dimensions '[d, rs ++ '[1] ++ ls]
  => All KnownNat '[n, ix]
  => All KnownDim '[n, ix]
  => Length d > ix ~ True
  => '(rs, n:+ls) ~ (SplitAt ix d)
  => Tensor d
  -> Dim ix
  -> KeepDim
  -> (Tensor (rs ++ '[1] ++ ls), Maybe (IndexTensor (rs ++ '[1] ++ ls)))
max = withKeepDim Dynamic._max

-- | Static call to 'Dynamic.min'
min
  :: forall d n ix rs ls
  .  All Dimensions '[d, rs ++ '[1] ++ ls]
  => All KnownNat '[n, ix]
  => All KnownDim '[n, ix]
  => Length d > ix ~ True
  => '(rs, n:+ls) ~ (SplitAt ix d)
  => Tensor d
  -> Dim ix
  -> KeepDim
  -> (Tensor (rs ++ '[1] ++ ls), Maybe (IndexTensor (rs ++ '[1] ++ ls)))
min = withKeepDim Dynamic._min

-- | Static call to 'Dynamic.median'
median
  :: forall d n ix rs ls
  .  All Dimensions '[d, rs ++ '[1] ++ ls]
  => All KnownNat '[n, ix]
  => All KnownDim '[n, ix]
  => Length d > ix ~ True
  => '(rs, n:+ls) ~ (SplitAt ix d)
  => Tensor d
  -> Dim ix
  -> KeepDim
  -> (Tensor (rs ++ '[1] ++ ls), Maybe (IndexTensor (rs ++ '[1] ++ ls)))
median = withKeepDim Dynamic._median

-- -- | Convenience method for 'max'
-- max1d :: (KnownDim n) => Tensor '[n] -> KeepDim -> (Tensor '[], Maybe (IndexTensor '[0]))
-- max1d t = Torch.Indef.Static.Tensor.Math.Reduce.max t (dim :: Dim 0)

-- -- | Convenience method for 'min'
-- min1d :: forall n . KnownDim n => Tensor '[n] -> KeepDim -> (Tensor '[n], Maybe (IndexTensor '[1]))
--   :: forall d n ix rs ls
--   .  All Dimensions '[d, rs ++ ls]
--   => All KnownNat '[n, ix]
--   => All KnownDim '[n, ix]
--   => '(rs, n:+ls) ~ (SplitAt ix d)
--   => Tensor '[n]
--   -> KeepDim
--   -> (Tensor '[], Maybe (IndexTensor '[1]))
-- min1d t = Torch.Indef.Static.Tensor.Math.Reduce.min t (dim :: Dim 0)

-- -- | Convenience method for 'median'
-- median1d
--   :: (KnownDim n)
--   => Tensor '[n] -> KeepDim -> (Tensor '[n], Maybe (IndexTensor '[1]))
-- median1d t = median t (dim :: Dim 0)

-- -- | Convenience method for 'max'
-- maxIndex1d t = fromJust . snd $ max1d t keep
--
-- -- | Convenience method for 'min'
-- minIndex1d t = fromJust . snd $ min1d t keep
--
-- -- | Convenience method for 'median'
-- medianIndex1d t = fromJust . snd $ median1d t keep

-- | Static call to 'Dynamic._prod'
_prod :: Tensor d -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_prod r t = Dynamic._prod (asDynamic r) (asDynamic t)

-------------------------------------------------------------------------------

withKeepDim
  :: forall d n ix rs ls
  .  All Dimensions '[d, rs ++ '[1] ++ ls]
  => All KnownNat '[n, ix]
  => All KnownDim '[n, ix]
  => Length d > ix ~ True
  => '(rs, n:+ls) ~ (SplitAt ix d)
  => ((Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ())
  -> Tensor d
  -> Dim ix
  -> KeepDim
  -> (Tensor (rs ++ '[1] ++ ls), Maybe (IndexTensor (rs ++ '[1] ++ ls)))
withKeepDim _fn t d k = unsafePerformIO $ do
  ret :: Tensor (rs ++ '[1] ++ ls) <- new
  let ix :: IndexTensor (rs ++ '[1] ++ ls) = newIx
  _fn (asDynamic ret, longAsDynamic ix) (asDynamic t) ((fromIntegral $ dimVal d)) (Just k)
  pure (ret, if coerce k then Just ix else Nothing)
{-# NOINLINE withKeepDim #-}


-- | Static call to 'Dynamic.sum'
sum :: Dimensions d' => Tensor d -> DimVal -> KeepDim -> Tensor d'
sum t d k = unsafePerformIO $ do
  r <- new
  Dynamic._sum (asDynamic r) (asDynamic t) d (Just k)
  pure r
{-# NOINLINE sum #-}

-- | convenience function for 'sum'
rowsum :: (All KnownDim '[r,c]) => Tensor '[r, c] -> (Tensor '[1, c])
rowsum t = Torch.Indef.Static.Tensor.Math.Reduce.sum t 0 keep

-- | convenience function for 'sum'
colsum :: (All KnownDim '[r,c]) => Tensor '[r, c] -> (Tensor '[r, 1])
colsum t = Torch.Indef.Static.Tensor.Math.Reduce.sum t 0 keep


