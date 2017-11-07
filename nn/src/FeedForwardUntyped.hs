{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Control.Exception.Base (assert)
import Data.Monoid ((<>))
import Data.Maybe (fromJust)
import Foreign.C.Types
import Foreign.Ptr

import TensorDouble
import TensorDoubleMath (td_sigmoid, td_addmv)
import TensorDoubleRandom
import TensorRaw
import Random
import TensorTypes
import TensorUtils

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

dispW w = do
  putStrLn "Biases:"
  disp (biases w)
  putStrLn "Weights:"
  disp (nodes w)

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
  dat <- td_uniform dat gen (-1.0) (1.0)
  pure dat

randomNet :: Word -> [Word] -> Word -> IO Network
randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> TensorDouble -> TensorDouble
runLayer (W wB wN) v = td_addmv 1.0 wB 1.0 wN v

runNet :: Network -> TensorDouble -> TensorDouble
runNet (O w) v = td_sigmoid (runLayer w v)
runNet (w :~ n') v = let v' = td_sigmoid (runLayer w v) in runNet n' v'


main = do
  net <- randomNet 5 [3, 2, 4, 2, 3] 2
  dat <- randomData 5
  putStrLn "Data\n--------"
  disp dat
  putStrLn "Network\n--------"
  dispN net
  let result = runNet net dat
  putStrLn "Result\n--------"
  disp result
  putStrLn "Done"
