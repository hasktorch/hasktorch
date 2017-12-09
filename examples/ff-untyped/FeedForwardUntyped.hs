{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleMath (td_sigmoid, td_addmv)
import Torch.Core.Tensor.Dynamic.DoubleRandom
import Torch.Core.Tensor.Types

data Weights = W {
  biases :: TensorDouble,
  nodes :: TensorDouble
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
  td_p (biases w)
  putStrLn "Weights:"
  td_p (nodes w)

dispN :: Network -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "Current Layer ::::\n" >> dispW w >> dispN n'

randomWeights :: Word -> Word -> IO Weights
randomWeights i o = do
  gen <- newRNG
  let w1 = W { biases = td_new (D1 o), nodes = td_new (D2 (o, i)) }
  b <- td_uniform (biases w1) gen (-1.0) (1.0)
  w <- td_uniform (nodes w1) gen (-1.0) (1.0)
  pure W { biases = b, nodes = w }

randomData :: Word -> IO TensorDouble
randomData i = do
  gen <- newRNG
  let dat = td_new (D1 i)
  td_uniform dat gen (-1.0) (1.0)

randomNet :: Word -> [Word] -> Word -> IO Network
randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> TensorDouble -> TensorDouble
runLayer (W wB wN) v = td_addmv 1.0 wB 1.0 wN v

runNet :: Network -> TensorDouble -> TensorDouble
runNet (O w) v = td_sigmoid (runLayer w v)
runNet (w :~ n') v = let v' = td_sigmoid (runLayer w v) in runNet n' v'


main :: IO ()
main = do
  net <- randomNet 5 [3, 2, 4, 2, 3] 2
  dat <- randomData 5
  putStrLn "Data\n--------"
  td_p dat
  putStrLn "Network\n--------"
  dispN net
  let result = runNet net dat
  putStrLn "Result\n--------"
  td_p result
  putStrLn "Done"
