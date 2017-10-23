{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}

module Main where

-- experimental AD implementation

import Control.Exception.Base (assert)
import Data.Monoid ((<>))
import Data.Maybe (fromJust)
import Foreign.C.Types
import Foreign.Ptr

import GHC.TypeLits (Nat, KnownNat, natVal)

import StaticTensorDouble
import TensorDouble
--import TensorDoubleMath (sigmoid, (!*), addmv)
import TensorDoubleRandom
import Random
import TensorTypes
import TensorUtils

{- Statically Typed Implementation -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights i o = SW {
  biases :: TDS 1 '[o],
  nodes :: TDS 2 '[i, o]
  } deriving (Show)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: SW i o -> StaticNetwork i '[] o
  (:~) :: (KnownNat h) => SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :~

dispW :: (KnownNat o, KnownNat i) => StaticWeights i o -> IO ()
dispW w = do
  putStrLn "Biases:"
  dispS (biases w)
  putStrLn "Weights:"
  dispS (nodes w)

-- dispN (O w) = dispW w
-- dispN (w :~ n') = putStrLn "Current Layer ::::\n" >> dispW w >> dispN n'

-- randomWeights :: Word -> Word -> IO (SW i o)
-- randomWeights i o = do
--   gen <- newRNG
--   let w1 = SW { biases = tdNew (D1 o), nodes = tdNew (D2 o i) }
--   b <- td_uniform (biases w1) gen (-1.0) (1.0)
--   w <- td_uniform (nodes w1) gen (-1.0) (1.0)
--   pure SW { biases = b, nodes = w }

-- randomData :: Word -> IO TensorDouble
-- randomData i = do
--   gen <- newRNG
--   let dat = tdNew (D1 i)
--   dat <- td_uniform dat gen (-1.0) (1.0)
--   pure dat

-- randomNet :: Word -> [Word] -> Word -> IO (SN i h o)
-- randomNet i [] o = O <$> randomWeights i o
-- randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

-- runLayer :: SW i o -> TensorDouble -> TensorDouble
-- runLayer (SW wB wN) v = addmv 1.0 wB 1.0 wN v

-- runNet :: SN i h o -> TensorDouble -> TensorDouble
-- runNet (O w) v = sigmoid (runLayer w v)
-- runNet (w :~ n') v = let v' = sigmoid (runLayer w v) in runNet n' v'

-- train :: Double
--       -> TensorDouble
--       -> TensorDouble
--       -> SN i h o
--       -> SN i h o
-- train rate x0 target = fst . go x0
--   where go x (O w@(SW wB wN)) = undefined



main = do
    putStrLn "Done"
