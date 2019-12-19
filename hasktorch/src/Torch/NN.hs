{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DeriveGeneric #-}

module Torch.NN where

import Control.Monad.State.Strict
import System.IO.Unsafe (unsafePerformIO)

import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import Torch.Internal.Cast (cast3)

import Torch.Autograd
import Torch.Initializers
import Torch.Tensor
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functional
import GHC.Generics

type Parameter = IndependentTensor
type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, Parameterized' (Rep f)) => f -> [Parameter]
  flattenParameters f = flattenParameters' (from f)

  replaceOwnParameters :: f -> ParamStream f
  default replaceOwnParameters :: (Generic f, Parameterized' (Rep f)) => f -> ParamStream f
  replaceOwnParameters f = to <$> replaceOwnParameters' (from f)

instance Parameterized Parameter where
  flattenParameters = pure
  replaceOwnParameters _ = nextParameter

instance Parameterized Double where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized [Int] where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized (Tensor -> Tensor) where
  flattenParameters _ = []
  replaceOwnParameters = return

class Parameterized' f where
  flattenParameters' :: f a -> [Parameter]
  replaceOwnParameters' :: f a -> ParamStream (f a)

instance Parameterized' U1 where
  flattenParameters' U1 = []
  replaceOwnParameters' U1 = return U1

instance (Parameterized' f, Parameterized' g) => Parameterized' (f :+: g) where
  flattenParameters' (L1 x) = flattenParameters' x
  flattenParameters' (R1 x) = flattenParameters' x
  replaceOwnParameters' (L1 x) = do
    x' <- replaceOwnParameters' x
    return $ L1 x'
  replaceOwnParameters' (R1 x) = do
    x' <- replaceOwnParameters' x
    return $ R1 x'

instance (Parameterized' f, Parameterized' g) => Parameterized' (f :*: g) where
  flattenParameters' (x :*: y) = flattenParameters' x ++ flattenParameters' y
  replaceOwnParameters' (x :*: y) = do
    x' <- replaceOwnParameters' x
    y' <- replaceOwnParameters' y
    return $ x' :*: y'

instance (Parameterized c) => Parameterized' (K1 i c) where
  flattenParameters' (K1 x) = flattenParameters x
  replaceOwnParameters' (K1 x) = do
    x' <- replaceOwnParameters x
    return $ K1 x'

instance (Parameterized' f) => Parameterized' (M1 i t f) where
  flattenParameters' (M1 x) = flattenParameters' x
  replaceOwnParameters' (M1 x) = do
    x' <- replaceOwnParameters' x
    return $ M1 x'

replaceParameters :: Parameterized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (replaceOwnParameters f) params in
  if null remaining
    then f'
    else error "Some parameters in a call to replaceParameters haven't been consumed!"

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

class (Randomizable spec f, Parameterized f) => Module spec f

data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
  deriving (Show, Eq)

data Linear = Linear { weight :: Parameter, bias :: Parameter } deriving (Show, Generic)

linear :: Linear -> Tensor -> Tensor
linear layer input = linear' input w b
    where
        linear' input weight bias = unsafePerformIO $ (cast3 ATen.linear_ttt) input weight bias
        w = toDependent (weight layer)
        b = toDependent (bias layer)

instance Randomizable LinearSpec Linear where
  sample LinearSpec{..} = do
      w <- makeIndependent =<< kaimingUniform' [out_features, in_features]
      -- w <- makeIndependent =<< randn' [out_features, in_features]
      b <- makeIndependent =<< randn' [out_features]
      return $ Linear w b

instance Parameterized Linear
-- This instance generates following codes.
--
---------------------------------------------------
-- instance Parameterized Linear where
--   flattenParameters Linear{..} = [weight, bias]
--   replaceOwnParameters _ = do
--     weight <- nextParameter
--     bias <- nextParameter
--     return $ Linear{..}

instance Parameterized [Linear]

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = map toDependent parameters
