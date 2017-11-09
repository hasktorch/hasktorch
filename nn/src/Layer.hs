{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MultiParamTypeClasses #-}

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
import Data.Singletons.TypeLits

-- datakinds lifts constructors to type level
data Shape = L1 Nat | L2 Nat Nat | L3 Nat Nat Nat | L4 Nat Nat Nat Nat

-- Define shape types
data S (shapeType :: Shape) where
  S1L :: (KnownNat o)                         => TDS '[o]          -> S ('L1 o)
  S2L :: (KnownNat r, KnownNat c)             => TDS '[r, c]       -> S ('L2 r c)
  S3L :: (KnownNat x, KnownNat y, KnownNat z) => TDS '[x, y, z]    -> S ('L3 x y z)
  S4L :: (KnownNat a, KnownNat b,
          KnownNat c, KnownNat d)             => TDS '[a, b, c, d] -> S ('L4 a b c d)

-- define singleton constructors: take type inputs, return value of type Sing [Shape]
data instance Sing (shapeType :: Shape) where
  L1Sing :: KnownNat a                                       => Sing ('L1 a)
  L2Sing :: (KnownNat a, KnownNat b)                         => Sing ('L2 a b)
  L3Sing :: (KnownNat a, KnownNat b, KnownNat c)             => Sing ('L3 a b c)
  L4Sing :: (KnownNat a, KnownNat b, KnownNat c, KnownNat d) => Sing ('L4 a b c d)

-- derive value-level singleton from layer shape types (L1...L4)
instance KnownNat a => SingI ('L1 a) where
  sing = L1Sing
instance (KnownNat a, KnownNat b) => SingI ('L2 a b) where
  sing = L2Sing
instance (KnownNat a, KnownNat b, KnownNat c) => SingI ('L3 a b c) where
  sing = L3Sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat d) => SingI ('L4 a b c d) where
  sing = L4Sing

-- typeclass 1: layers can be updated and randomly instantiated
class UpdateLayer layer where
  type Gradient layer :: *
  updateLayer :: LearningParameters -> layer -> Gradient layer -> layer
  createRandom :: IO layer

-- typeclass 2: layers can be run forward and backward propagated
class UpdateLayer x => Layer x (i :: Shape) (o :: Shape) where
  type Tape x i o :: *
  runForwards :: x -> S i -> (Tape x i o, S o)
  runBackwards :: x -> (Tape x i o, S o) -> S o -> (Gradient x, S i)

-- learning parameter values
data LearningParameters = LearningParameters {
  lp_rate :: Double
  , lp_momentum :: Double
  , lp_regularizer :: Double
  } deriving (Eq, Show)

-- a network is a list of layers
data Network :: [*] -> [Shape] -> * where
  NNil :: SingI i => Network '[] '[i]
  NCons :: (SingI i, SingI h, Layer x i h) =>
    x -> (Network xs (h ': hs)) -> Network (x ': xs) (i ': h ': hs)
