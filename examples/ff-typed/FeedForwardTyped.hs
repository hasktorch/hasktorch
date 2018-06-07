{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
module Main where

import Numeric.Dimensions
import System.IO.Unsafe (unsafePerformIO)

import Torch.Double
{- Simple FF neural network, statically typed version, based on JL's example -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights (i :: Nat) (o :: Nat) = SW
  { biases :: DoubleTensor '[o]
  , nodes :: DoubleTensor '[o, i]
  } deriving (Show)

mkW :: (KnownDim i, KnownDim o) => IO (SW i o)
mkW = SW <$> new <*> new

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownDim i, KnownDim o) =>
       SW i o -> SN i '[] o
  (:~) :: (KnownDim h, KnownDim i, KnownDim o) =>
          SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :~

dispW :: (KnownDim o, KnownDim i) => StaticWeights i o -> IO ()
dispW w = do
  putStrLn "\nBiases:"
  print (biases w)
  putStrLn "\nWeights:"
  print (nodes w)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispW w >> dispN n'

randomWeights :: (KnownDim i, KnownDim o) => IO (SW i o)
randomWeights = do
  gen <- newRNG
  let Just bs = ord2Tuple (-1, 1)
  b <- uniform gen bs
  w <- uniform gen bs
  pure SW { biases = b, nodes = w }

randomNet :: forall i hs o. (Dimensions hs, KnownDim i, KnownDim o) => IO (SN i hs o)
randomNet = go (dims :: Dims hs)
  where
    go :: forall h hs'. KnownDim h => Dims hs' -> IO (SN h hs' o)
    go = \case
      Empty         ->    O <$> randomWeights
      Dim `Cons` ss -> (:~) <$> randomWeights <*> go ss

runLayer :: (KnownDim i, KnownDim o) => SW i o -> DoubleTensor '[i] -> IO (DoubleTensor '[o])
runLayer sw v = pure $ addmv 1.0 wB 1.0 wN v -- v are the inputs
  where wB = biases sw
        wN = nodes sw

runNet :: (KnownDim i, KnownDim o) => SN i hs o -> DoubleTensor '[i] -> IO (DoubleTensor '[o])
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
  runNet n1 (constant 1 :: DoubleTensor '[4]) >>= print

  putStrLn "\n=========\nNETWORK 2\n========="
  n2  <- randomNet :: IO (SN 4 '[3, 2] 2)
  dispN n2

  putStrLn "\nNETWORK 2 Forward prop result:"
  runNet n2 (constant 1 :: DoubleTensor '[4]) >>= print

  putStrLn "Done"
