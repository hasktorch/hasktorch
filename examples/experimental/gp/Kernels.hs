{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Kernels where

import Data.List (tails)
import Prelude as P
import Torch.Double as T
import qualified Torch.Core.Random as RNG

{- Helper functions -}

sum' :: KnownDim d => Tensor '[d] -> Tensor '[1]
sum' x = T.sum x 0 (KeepDim True)
-- FIXME make T.sum dimension safe in the core library

{- Kernels - 1 observation, d-dimensions -}

kernel_rbf :: KnownDim d =>
    Double -> Double -> Tensor '[d] -> Tensor '[d] -> Tensor '[1]
kernel_rbf (sigma :: Double) (length :: Double) t t' =
    (sigma^2) *^ T.exp eterm
    where
        eterm = (sum' $ - (t - t')^2) ^/ (2 * length^2)

kernel_periodic :: KnownDim d =>
    Double -> Double -> Double -> Tensor '[d] -> Tensor '[d] -> Tensor '[1]
kernel_periodic (sigma :: Double) (length :: Double) (period :: Double) t t' =
    (sigma^2) *^ (2 * (T.sin $ pi * (sum' (T.abs $ t - t')) ^/ period))
      ^/ length^2
    where pi' = scalar pi

kernel_linear :: KnownDim d =>
     Double -> Double -> Double -> Tensor '[d] -> Tensor '[d] -> Tensor '[1]
kernel_linear (sigma :: Double) (sigma_b :: Double) (c :: Double) t t' =
    (sigma_b^2) +^ ((sigma^2) *^ (sum' $ (t ^- c) ^*^ (t' ^- c)))

{- Kernels - 1 dimension, d observations -}

kernel1d_rbf :: KnownDim d =>
    Double -> Double -> Tensor '[d] -> Tensor '[d] -> Tensor '[d]
kernel1d_rbf (sigma :: Double) (length :: Double) t t' =
    (sigma^2) *^ T.exp eterm
    where
        eterm = (-(t - t')^2) ^/ (2 * length^2)