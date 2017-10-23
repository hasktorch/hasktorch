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

import TensorDouble
import TensorDoubleMath (sigmoid, (!*), addmv)
import TensorDoubleRandom
import TensorRaw
import Random
import TensorTypes
import TensorUtils

data Weights = W {
  biases :: TensorDouble,
  nodes :: TensorDouble
  } deriving (Eq, Show)

{- Dynamically Typed Implementation -}

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
  let w1 = W { biases = tdNew (D1 o), nodes = tdNew (D2 o i) }
  b <- td_uniform (biases w1) gen (-1.0) (1.0)
  w <- td_uniform (nodes w1) gen (-1.0) (1.0)
  pure W { biases = b, nodes = w }

randomData :: Word -> IO TensorDouble
randomData i = do
  gen <- newRNG
  let dat = tdNew (D1 i)
  dat <- td_uniform dat gen (-1.0) (1.0)
  pure dat

randomNet :: Word -> [Word] -> Word -> IO Network
randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> TensorDouble -> TensorDouble
runLayer (W wB wN) v = addmv 1.0 wB 1.0 wN v

runNet :: Network -> TensorDouble -> TensorDouble
runNet (O w) v = sigmoid (runLayer w v)
runNet (w :~ n') v = let v' = sigmoid (runLayer w v) in runNet n' v'

train :: Double
      -> TensorDouble
      -> TensorDouble
      -> Network
      -> Network
train rate x0 target = fst . go x0
  where go x (O w@(W wB wN)) = undefined

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
