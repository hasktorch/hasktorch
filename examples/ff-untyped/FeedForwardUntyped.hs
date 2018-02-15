{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE BangPatterns #-}
module Main where

import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Dynamic
import Torch.Core.Tensor.Dynamic.Math -- (sigmoid, addmv)
import Torch.Core.Tensor.Dynamic.Random

data Weights = W
  { biases :: DoubleTensor
  , nodes :: DoubleTensor
  } deriving (Eq, Show)

{- Simple FF neural network, dynamically typed version, based on JL's example -}

data Network :: * where
  O :: Weights -> Network
  (:~) :: Weights -> Network -> Network
  deriving Show

infixr 5 :~

dispW :: Weights -> IO ()
dispW w = do
  putStrLn "Biases:"
  printTensor (biases w)
  putStrLn "Weights:"
  printTensor (nodes w)

dispN :: Network -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "Current Layer ::::\n" >> dispW w >> dispN n'

randomWeights :: Word -> Word -> IO Weights
randomWeights i o = do
  gen <- newRNG
  d1 <- someDimsM [fromIntegral o]
  d2 <- someDimsM [fromIntegral o, fromIntegral i]
  w1 <- W <$> new' d1 <*> new' d2
  b <- uniform (biases w1) gen (-1.0) (1.0)
  w <- uniform (nodes w1) gen (-1.0) (1.0)
  pure W { biases = b, nodes = w }

randomData :: Word -> IO DoubleTensor
randomData i = do
  gen <- newRNG
  someD1 <- someDimsM [fromIntegral i]
  dat <- new' someD1
  uniform dat gen (-1.0) (1.0)

randomNet :: Word -> [Word] -> Word -> IO Network
randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> DoubleTensor -> IO DoubleTensor
runLayer (W wB wN) v = addmv 1.0 wB 1.0 wN v

runNet :: Network -> DoubleTensor -> IO DoubleTensor
runNet (O w) v     = (runLayer w v) >>= sigmoid
runNet (w :~ n') v = (runLayer w v) >>= sigmoid >>= runNet n'


main :: IO ()
main = do
  net <- randomNet 5 [3, 2, 4, 2, 3] 2
  dat <- randomData 5
  putStrLn "Data\n--------"
  printTensor dat
  putStrLn "Network\n--------"
  dispN net
  let result = runNet net dat
  putStrLn "Result\n--------"
  printTensor result
  putStrLn "Done"
