{-# LANGUAGE DataKinds, GADTs, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Layer where

import StaticTensorDouble
import StaticTensorDoubleMath
import StaticTensorDoubleRandom
import TensorDouble
import Random
import TensorTypes
import TensorUtils

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.Prelude.List
import Data.Singletons.TypeLits

data S (d :: [Nat]) where
  S :: TDS d -> S d

lmap :: forall d . (SingI d) => (TDS d -> TDS d) -> S d -> S d
lmap f (S t) = S (f t)

-- typeclass 1: layers can be updated and randomly instantiated
class UpdateLayer layer where
  type Gradient layer :: *
  updateLayer :: LearningParameters -> layer -> Gradient layer -> layer
  createRandom :: IO layer

-- typeclass 2: layers can be run forward and backward propagated
class UpdateLayer x => Layer x (i :: [Nat]) (o :: [Nat]) where
  type Tape x i o :: *
  runForwards :: x -> S i -> (Tape x i o, S o)
  runBackwards :: x -> Tape x i o -> S o -> (Gradient x, S i)

-- learning parameter values
data LearningParameters = LearningParameters {
  lp_rate :: Double
  , lp_momentum :: Double
  , lp_regularizer :: Double
  } deriving (Eq, Show)

-- a network is a list of layers
data Network :: [*] -> [[Nat]] -> * where
  NNil :: SingI i => Network '[] '[i]
  NCons :: (SingI i, SingI h, Layer x i h) =>
    x -> (Network xs (h : hs)) -> Network (x : xs) (i : h : hs)

--
-- Layer definiions
--

{- Logit -}

data Logit = Logit
  deriving Show

instance UpdateLayer Logit where
  type Gradient Logit = ()
  updateLayer _ _ _ = Logit
  createRandom = return Logit

instance (a ~ b, SingI a) => Layer Logit a b where
  type Tape Logit a b = S a
  runForwards _ a = (a, undefined)
  runBackwards _ a g = ((), undefined)
  -- runForwards _ a = (a, logistic a)
  -- runBackwards _ a g = ((), logistic' a * g)

-- -- logistic :: Floating a => a -> a
-- -- logistic x = 1 / (1 + exp (-x))
-- -- logistic' x = (logistic x) * (1 - (logistic x))

-- {- Tanh -}

data Tanh = Tanh
  deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  updateLayer _ _ _ = Tanh
  createRandom = return Tanh

instance (a ~ b, SingI a) => Layer Tanh a b where
  type Tape Tanh a b = S a
  runForwards _ a = (a, undefined)
  runBackwards _ a g = ((), undefined)
  -- runForwards _ a = (a, tanh a)
  -- runBackwards _ a g = ((), tanh' a * g)
