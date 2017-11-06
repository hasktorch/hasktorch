{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}

module Main where

-- experimental AD implementation

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

{- Statically Typed Implementation -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights (i :: Nat) (o :: Nat) = SW {
  biases :: TDS '[o],
  nodes :: TDS '[o, i]
  } deriving (Show)

mkW :: (KnownNat i, KnownNat o) => SW i o
mkW = SW biases nodes
  where (biases, nodes) = (tds_new, tds_new)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       SW i o -> SN i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :~

dispW :: (KnownNat o, KnownNat i) => StaticWeights i o -> IO ()
dispW w = do
  putStrLn "\nBiases:"
  dispS (biases w)
  putStrLn "\nWeights:"
  dispS (nodes w)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispW w >> dispN n'

randomWeights :: (KnownNat i, KnownNat o) => IO (SW i o)
randomWeights = do
  gen <- newRNG
  b <- tds_uniform (biases storeResult) gen (-1.0) (1.0)
  w <- tds_uniform (nodes storeResult) gen (-1.0) (1.0)
  pure SW { biases = b, nodes = w }
  where
    storeResult = mkW

randomNet :: forall i hs o. (KnownNat i, SingI hs, KnownNat o) => IO (SN i hs o)
randomNet = go (sing :: Sing hs)
  where go :: forall h hs'. KnownNat h => Sing hs' -> IO (SN h hs' o)
        go = \case
          SNil ->
            O <$> randomWeights
          SNat `SCons` ss ->
            (:~) <$> randomWeights <*> go ss

runLayer :: (KnownNat i, KnownNat o) => SW i o -> TDS '[i] -> TDS '[o]
runLayer sw v = tds_addmv 1.0 wB 1.0 wN v -- v are the inputs
  where wB = biases sw
        wN = nodes sw

runNet :: (KnownNat i, KnownNat o) => SN i hs o -> TDS '[i] -> TDS '[o]
runNet (O w) v = tds_sigmoid (runLayer w v)
runNet (w :~ n') v = let v' = tds_sigmoid (runLayer w v) in runNet n' v'

ih = mkW :: StaticWeights 10 7
hh = mkW :: StaticWeights  7 4
ho = mkW :: StaticWeights  4 2

net1 = O ho :: SN 4 '[] 2
net2 = hh :~ O ho :: SN 7 '[4] 2
net3 = ih :~ hh :~ O ho :: SN 10 '[7,4] 2

main = do
  putStrLn "\n=========\nNETWORK 1\n========="
  n1 <- (randomNet :: IO (SN 4 '[] 2))
  dispN n1

  putStrLn "\nNETWORK 1 Forward prop result:"
  dispS $ runNet n1 (tds_init 1.0 :: TDS '[4])

  putStrLn "\n=========\nNETWORK 2\n========="
  n2  <- randomNet :: IO (SN 4 '[3, 2] 2)
  dispN n2

  putStrLn "\nNETWORK 2 Forward prop result:"
  dispS $ runNet n2 (tds_init 1.0 :: TDS '[4])

  putStrLn "Done"
