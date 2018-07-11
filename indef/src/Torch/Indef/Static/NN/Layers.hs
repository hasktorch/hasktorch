-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Layers
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Miscellaneous layer functions.
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
module Torch.Indef.Static.NN.Layers where

import Data.List
import Data.Singletons.Prelude.List hiding (All)
import Numeric.Backprop
import Numeric.Dimensions

import Debug.Trace as D
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Reduce
import Torch.Indef.Static.Tensor.Math.Pairwise ((^/), (^-))
import Torch.Indef.Static.Tensor.Math.Pointwise ((^*^), (^-^))
import Torch.Indef.Static.Tensor.Math.Blas
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- | A backpropable 'flatten' operation
flattenBP
  :: (Reifies s W, KnownDim (Product d), Dimensions (d::[Nat]))
  => BVar s (Tensor d) -> BVar s (Tensor '[Product d])
flattenBP = liftOp1 . op1 $ \t -> (flatten t, resizeAs)


-------------------------------------------------------------------------------

-- |  sparseLinear forward pass (updates the output tensor)
_sparseLinear_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_updateOutput t0 t1 t2 t3 = Dynamic._sparseLinear_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
-- |  sparseLinear backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_sparseLinear_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
_sparseLinear_accGradParameters t0 t1 t2 t3 t4 t5 = Dynamic._sparseLinear_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

-- |  sparseLinear zeroGradParameters
_sparseLinear_zeroGradParameters :: Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_zeroGradParameters t0 t1 t2 = Dynamic._sparseLinear_zeroGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  sparseLinear updateParameters
_sparseLinear_updateParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_sparseLinear_updateParameters t0 t1 t2 t3 t4 = Dynamic._sparseLinear_updateParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

-- |  gatedLinear forward pass (updates the output tensor)
_gatedLinear_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateOutput t0 t1 = Dynamic._gatedLinear_updateOutput (asDynamic t0) (asDynamic t1)
-- |  gatedLinear backward-update (updates the layer and bias tensors)
_gatedLinear_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateGradInput t0 t1 t2 = Dynamic._gatedLinear_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  gRUFused forward pass (updates the output tensor)
_gRUFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._gRUFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
-- |  gRUFused backward-update (updates the layer and bias tensors)
_gRUFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateGradInput t0 t1 t2 t3 t4 = Dynamic._gRUFused_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

-- |  lSTMFused forward pass (updates the output tensor)
_lSTMFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
-- |  lSTMFused backward-update (updates the layer and bias tensors)
_lSTMFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateGradInput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)


