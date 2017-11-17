{-# LANGUAGE DataKinds, GADTs, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}

module Layer where

import StaticTensorDouble
import StaticTensorDoubleMath
import StaticTensorDoubleRandom
import TensorDouble
import Random
import TensorTypes

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.Prelude.List
import Data.Singletons.TypeLits

--
-- Layer representation
--

-- each layer is associated with a single tensor of arbitrary dimension
data S (d :: [Nat]) where
  S :: TDS d -> S d

-- class constraint for a type-level list of shapes
class KnownShapes (ns :: [[Nat]]) where
  natsVal  :: p ns -> [[Integer]]

-- fmap-like application to layer applies it to the associated tensor
lmap :: forall d . (SingI d) => (TDS d -> TDS d) -> S d -> S d
lmap f (S t) = S (f t)

-- layers can be updated and randomly instantiated
class UpdateLayer l where
  type Gradient l :: *
  updateLayer :: LearningParameters -> l -> Gradient l -> l
  createRandom :: IO l

-- layers can be run forward and backward propagated
class UpdateLayer l => Layer l (i :: [Nat]) (o :: [Nat]) where
  type Tape oper i o :: *
  runForwards :: l -> S i -> (Tape l i o, S o)
  runBackwards :: l -> Tape l i o -> S o -> (Gradient l, S i)

--
-- Network representation
--

-- a network is a list of layer shapes
data Network :: [*] -> [[Nat]] -> * where
  NNil :: SingI i => Network '[] '[i]
  (:~) :: (SingI i, SingI h, Layer x i h) =>
    x -> (Network xs (h ': hs)) -> Network (x ': xs) (i ': h ': hs)

-- show instance for networks
instance Show (Network '[] '[i]) where
  show NNil = "NNil"
instance (Show x, Show (Network xs rs)) => Show (Network (x ': xs) (i ': rs)) where
  show (x :~ xs) = show x ++ "\n~\n" ++ show xs

runNetwork :: forall layers shapes . KnownShapes shapes =>
              Network layers shapes -> S (Head shapes)
           -> (Tapes layers shapes, S (Last shapes))
runNetwork = go
  where
    go  :: forall js ss. (Last js ~ Last shapes)
        => Network ss js -> S (Head js) -> (Tapes ss js, S (Last js))
    go (layer :~ n) !x = (tape :\> tapes, answer)
      where (tape, forward) = runForwards layer x
            (tapes, answer) = go n forward
    go NNil x = (TNil, x)

runGradient :: forall layers shapes . KnownShapes shapes =>
               Network layers shapes -> Tapes layers shapes -> S (Last shapes)
            -> (Gradients layers, S (Head shapes))
runGradient net tapes o = go net tapes
  where
    go :: forall js ss. (Last js ~ Last shapes) =>
        Network ss js -> Tapes ss js -> (Gradients ss, S (Head js))
    go (layer :~ n) (tape :\> nt) = (layer' :/> gradients, backGrad)
      where (gradients, feed)  = go n nt
            (layer', backGrad) = runBackwards layer tape feed
    go NNil TNil = (GNil, o)

data Gradients :: [*] -> * where
   GNil  :: Gradients '[]
   (:/>) :: UpdateLayer x =>
            Gradient x -> Gradients xs -> Gradients (x : xs)

data Tapes :: [*] -> [[Nat]] -> * where
   TNil  :: SingI i => Tapes '[] '[i]
   (:\>) :: (SingI i, SingI h, Layer x i h) =>
         (Tape x i h) -> (Tapes xs (h : hs)) -> Tapes (x : xs) (i : h : hs)

--
-- Learning parameters
--

-- learning parameter values
data LearningParameters = LearningParameters {
  lp_rate :: Double
  , lp_momentum :: Double
  , lp_regularizer :: Double
  } deriving (Eq, Show)

--
-- Layer definiions
--

{- Logit -}

data Logit = Logit deriving Show

instance UpdateLayer Logit where
  type Gradient Logit = ()
  updateLayer _ _ _ = Logit
  createRandom = return Logit

instance (din ~ dout, SingI din) => Layer Logit din dout where
  type Tape Logit din dout = S din
  runForwards _ input@(S it) = (input, S (tds_sigmoid it))
  runBackwards _ input@(S it) grad@(S ot) = ((), S $ tds_cmul (tds_sigmoid' it) ot)

tds_sigmoid' t = tds_cmul (tds_sigmoid t) (1.0 -^ (tds_sigmoid t))

{- Tanh -}

data Tanh = Tanh deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  updateLayer _ _ _ = Tanh
  createRandom = return Tanh

instance (din ~ dout, SingI din) => Layer Tanh (din :: [Nat]) (dout :: [Nat]) where
  type Tape Tanh din dout = S din
  runForwards _ input@(S it) = (input, S (tds_tanh it))
  runBackwards _ input@(S it) grad@(S ot) = ((), S (tds_cmul (tanh' it) ot))

tanh' t = 1.0 -^ (tds_pow s 2.0)
  where s = tds_tanh t

{- Relu -}

data Relu = Relu deriving Show

instance UpdateLayer Relu where
  type Gradient Relu = ()
  updateLayer _ _ _ = Relu
  createRandom = return Relu

instance (din ~ dout, SingI din) => Layer Relu (din :: [Nat]) (dout :: [Nat]) where
  type Tape Relu din dout = S din
  runForwards _ (S y) = (S y, S(relu y))
    where
      relu t = undefined
  runBackwards _ (S y) (S dEdy) = ((), S (tds_cmul (relu' y) dEdy))
    where
      relu' t = undefined

{- Tests -}

type Net0 = Network '[ ]             '[ ]
type Net1 = Network '[ Logit ]       '[ '[1] ]
type Net2 = Network '[ Logit, Tanh ] '[ '[1], '[5] ]
