{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math
import qualified Torch.Core.Random as RNG
import System.IO.Unsafe (unsafePerformIO)

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

{- Simple FF neural network, statically typed version, based on JL's example -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights (i :: Nat) (o :: Nat) = SW
  { biases :: DoubleTensor '[o]
  , nodes :: DoubleTensor '[o, i]
  } deriving (Show)

mkW :: (KnownNatDim i, KnownNatDim o) => IO (SW i o)
mkW = SW <$> new <*> new

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNatDim i, KnownNatDim o) =>
       SW i o -> SN i '[] o
  (:~) :: (KnownNatDim h, KnownNatDim i, KnownNatDim o) =>
          SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :~

dispW :: (KnownNatDim o, KnownNatDim i) => StaticWeights i o -> IO ()
dispW w = do
  putStrLn "\nBiases:"
  printTensor (biases w)
  putStrLn "\nWeights:"
  printTensor (nodes w)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispW w >> dispN n'

randomWeights :: (KnownNatDim i, KnownNatDim o) => IO (SW i o)
randomWeights = do
  gen <- RNG.new
  b <- uniform gen (-1.0) (1.0)
  w <- uniform gen (-1.0) (1.0)
  pure SW { biases = b, nodes = w }

randomNet :: forall i hs o. (KnownNatDim i, SingI hs, KnownNatDim o) => IO (SN i hs o)
randomNet = go (sing :: Sing hs)
  where
    go :: forall h hs'. KnownNatDim h => Sing hs' -> IO (SN h hs' o)
    go = \case
      SNil            ->    O <$> randomWeights
      SNat `SCons` ss -> (:~) <$> randomWeights <*> go ss

runLayer :: (KnownNatDim i, KnownNatDim o) => SW i o -> DoubleTensor '[i] -> IO (DoubleTensor '[o])
runLayer sw v = addmv 1.0 wB 1.0 wN v -- v are the inputs
  where wB = biases sw
        wN = nodes sw

runNet :: (KnownNatDim i, KnownNatDim o) => SN i hs o -> DoubleTensor '[i] -> IO (DoubleTensor '[o])
runNet (O w) v = do
  l <- runLayer w v
  sigmoid l

runNet (w :~ n') v = do
  v' <- (runLayer w v)
  active <- sigmoid v'
  runNet n' active

ih :: StaticWeights 10 7
hh :: StaticWeights  7 4
ho :: StaticWeights  4 2
ih = unsafePerformIO mkW
{-# NOINLINE ih #-}
hh = unsafePerformIO mkW
{-# NOINLINE hh #-}
ho = unsafePerformIO mkW
{-# NOINLINE ho #-}

net1 :: SN 4 '[] 2
net1 = O ho
net2 :: SN 7 '[4] 2
net2 = hh :~ O ho

net3 :: SN 10 '[7,4] 2
net3 = ih :~ hh :~ O ho

main :: IO ()
main = do
  putStrLn "\n=========\nNETWORK 1\n========="
  n1 <- (randomNet :: IO (SN 4 '[] 2))
  dispN n1

  putStrLn "\nNETWORK 1 Forward prop result:"
  (constant 1 :: IO (DoubleTensor '[4])) >>= runNet n1 >>= printTensor

  putStrLn "\n=========\nNETWORK 2\n========="
  n2  <- randomNet :: IO (SN 4 '[3, 2] 2)
  dispN n2

  putStrLn "\nNETWORK 2 Forward prop result:"
  (constant 1 :: IO (DoubleTensor '[4])) >>= runNet n2 >>= printTensor

  putStrLn "Done"
