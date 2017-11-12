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
class UpdateLayer l where
  type Gradient l :: *
  updateLayer :: LearningParameters -> l -> Gradient l -> l
  createRandom :: IO l

-- typeclass 2: layers can be run forward and backward propagated
class UpdateLayer l => Layer l (i :: [Nat]) (o :: [Nat]) where
  type Tape oper i o :: *
  runForwards :: l -> S i -> (Tape l i o, S o)
  runBackwards :: l -> Tape l i o -> S o -> (Gradient l, S i)

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
  runForwards _ input@(S it) = (input, S (tds_sigmoid it))
  runBackwards _ input@(S it) grad@(S ot) = ((), S $ tds_cmul (tds_sigmoid' it) ot)

tds_sigmoid' t = tds_cmul (tds_sigmoid t) (1.0 -^ (tds_sigmoid t))

{- Tanh -}

data Tanh = Tanh
  deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  updateLayer _ _ _ = Tanh
  createRandom = return Tanh

instance (a ~ b, SingI a) => Layer Tanh a b where
  type Tape Tanh a b = S a
  runForwards _ input@(S it) = (input, S (tds_tanh it))
  runBackwards _ input@(S it) grad@(S ot) = ((), S (tds_cmul (tanh' it) ot))

tanh' t = 1.0 -^ (tds_pow s 2.0)
  where s = tds_tanh t
