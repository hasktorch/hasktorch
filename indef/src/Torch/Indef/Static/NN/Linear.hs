-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Linear
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Linear layers
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
module Torch.Indef.Static.NN.Linear where

import Data.List
import Data.Singletons.Prelude.List hiding (All)
import Numeric.Backprop
import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Blas
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- | datatype representing a linear layer with bias. Represents
-- @y = Ax + b@.
newtype Linear i o
  = Linear { getTensors :: (Tensor '[i, o], Tensor '[o]) }

instance (KnownDim i, KnownDim o) => Show (Linear i o) where
  show c = intercalate ","
    [ "Linear ("
    ++ "input: "  ++ show (inputSize c)
    , " output: " ++ show (outputSize c)
    ++ ")"
    ]

instance (KnownDim i, KnownDim o) => Backprop (Linear i o) where
  zero = const . Linear $ (constant 0, constant 0)
  one  = const . Linear $ (constant 1, constant 1)
  add c0 c1 = Linear (weights c0 + weights c1, bias c0 + bias c1)

-- | the dense weight matrix of a linear layer
weights :: Linear i o -> Tensor '[i, o]
weights (Linear (w, _)) = w

-- | the bias vector of a linear layer
bias :: Linear i o -> Tensor '[o]
bias (Linear (_, b)) = b

-- | The input size of a linear layer
inputSize :: forall i o . KnownDim i => Linear i o -> Int
inputSize _ = fromIntegral (dimVal (dim :: Dim i))

-- | The output size of a linear layer
outputSize :: forall i o kW dW . KnownDim o => Linear i o -> Int
outputSize _ = fromIntegral (dimVal (dim :: Dim o))

-- ========================================================================= --

-- | Backprop linear function without batching
linear
  :: forall s i o
  .  Reifies s W
  => All KnownDim '[i,o]
  => BVar s (Linear i o)
  -> BVar s (Tensor '[i])
  -> BVar s (Tensor '[o])
linear = liftOp2 $ op2 $ \l i -> (transpose2d (weights l) `mv` i + bias l, go l i)
  where
    go :: Linear i o -> Tensor '[i] -> Tensor '[o] -> (Linear i o, Tensor '[i])
    go (Linear (w, b)) i gout = (Linear (i `outer` b', b'), w `mv` b')
      where
        b' = gout - b


