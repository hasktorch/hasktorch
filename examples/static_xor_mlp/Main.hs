{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude hiding (tanh)
import Torch.Static
import Torch.Static.Native hiding (linear)
import Torch.Static.Factories
import Torch.DType
import qualified Torch.Autograd as A
import qualified Torch.NN as A
import qualified Torch.Tensor as D
import qualified Torch.Functions as D
import qualified Torch.TensorFactories as D
import GHC.Generics
import GHC.TypeLits
import Data.Reflection

import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------


data Parameter d s = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter d s -> Tensor d s
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t


instance A.Parameterized (Parameter d s) where
  flattenParameters (Parameter x) = [x]
  replaceOwnParameters _ = Parameter <$> A.nextParameter

data LinearSpec (d::DType) (i::Nat) (o::Nat) = LinearSpec
  deriving (Show, Eq)

data Linear (d::DType) (in_features::Nat) (out_features::Nat) =
  Linear { weight :: Parameter d '[in_features,out_features]
         , bias :: Parameter d '[out_features]
         } deriving (Show, Generic)

linear :: Linear d i o -> Tensor d '[k,i] -> Tensor d '[k,o]
linear Linear{..} input = add (mm input (toDependent weight)) (toDependent bias)

makeIndependent :: Tensor d s -> IO (Parameter d s)
makeIndependent t = Parameter <$> A.makeIndependent (toDynamic t)

instance A.Parameterized (Linear d n m)

instance (KnownDType d, KnownNat n, KnownNat m) => A.Randomizable (LinearSpec d n m) (Linear d n m) where
  sample LinearSpec = do
      w <- makeIndependent =<< (randn :: IO (Tensor d '[n,m]))
      b <- makeIndependent =<< (randn :: IO (Tensor d '[m]))
      return $ Linear w b

data MLPSpec (d::DType) (i::Nat) (o::Nat) = MLPSpec

data MLP (d::DType) (i::Nat) (o::Nat) =
  MLP { l0 :: Linear d i 20
      , l1 :: Linear d 20 20
      , l2 :: Linear d 20 o
      } deriving (Generic)

instance A.Parameterized (MLP d i o)

instance (KnownDType d, KnownNat n, KnownNat m) => A.Randomizable (MLPSpec d n m) (MLP d n m) where
  sample MLPSpec = do
      l0 <- A.sample LinearSpec :: IO (Linear d n 20)
      l1 <- A.sample LinearSpec :: IO (Linear d 20 20)
      l2 <- A.sample LinearSpec :: IO (Linear d 20 m)
      return $ MLP {..}

mlp :: MLP d i o -> Tensor d '[b,i] -> Tensor d '[b,o]
mlp MLP{..} input = linear l2 . tanh . linear l1 . tanh . linear l0 $ input

batch_size = 32
num_iters = 10000

model :: MLP 'Float 2 1 -> Tensor 'Float '[n,2] -> Tensor 'Float '[n,1]
model params t = sigmoid (mlp params t)

main = do
    init <- A.sample $ (MLPSpec :: MLPSpec 'Float 2 1)
    trained <- foldLoop init num_iters $ \state i -> do
        input <- D.rand' [batch_size, 2] >>= return . (D.toDType Float) . (D.gt 0.5)
        let expected_output = tensorXOR input

        let output = D.squeezeAll $ toDynamic $ model state (UnsafeMkTensor input :: Tensor 'Float '[batch_size,2])
        let loss = D.mse_loss output expected_output

        let flat_parameters = A.flattenParameters state
        let gradients = A.grad loss flat_parameters

        if i `mod` 100 == 0
          then do putStrLn $ show loss
          else return ()

        new_flat_parameters <- mapM A.makeIndependent $ A.sgd 5e-4 flat_parameters gradients
        return $ A.replaceParameters state $ new_flat_parameters
    return ()
  where
    foldLoop x count block = foldM block x [1..count]
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = D.select t 1 0
        b = D.select t 1 1
