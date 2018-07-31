-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Backprop
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Backprop helper instances for static tensors, as well as any helper
-- functions that might work well with backprop.
-------------------------------------------------------------------------------
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.NN.Backprop where

import Data.Singletons.Prelude.List hiding (All, Drop, type (++))
import Numeric.Backprop
import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed ()
import qualified Torch.Indef.Index as Ix
import qualified Torch.Indef.Static.Tensor as T


instance Dimensions d => Backprop (Tensor d) where
  add = (+)
  zero = (const . constant) 0
  one = (const . constant) 1

  -- :: Dimensions d
  -- => Dim n
  -- -> Tensor d
  -- -> Tensor (rs ++ '[1] ++ ls)

unsqueeze1dBP
  :: forall s d rs ls n
  .  Reifies s W
  => All Dimensions '[d, (rs ++ '[1] ++ ls)]
  => '( rs, ls) ~ (SplitAt n d)
  => '( rs, 1:+ls) ~ (SplitAt n (rs ++ '[1] ++ ls))
  => (rs ++ ls) ~ d
  => Dim n
  -> BVar s (Tensor d)
  -> BVar s (Tensor (rs ++ '[1] ++ ls))
unsqueeze1dBP d = liftOp1 . op1 $ \t ->
  (T.unsqueeze1d d t, go)
  where
    go :: Tensor (rs ++ '[1] ++ ls) -> Tensor d
    go o = T.squeeze1d d o

    -- d :: Dim 0
    -- d = dim
clip :: Reifies s W => (HsReal, HsReal) -> BVar s (Tensor '[1]) -> BVar s (Tensor '[1])
clip (mn,mx) = liftOp1 . op1 $ \i ->
  let
    x = case get1d i 0 of
          x | x > mx ->   mx
            | x < mn ->   mn
            | otherwise -> x
  in
    (scalar x, id)
