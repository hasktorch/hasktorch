{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}

module Main where

-- experimental AD implementation

import Control.Exception.Base (assert)
import Data.Monoid ((<>))
import Data.Maybe (fromJust)
import Foreign.C.Types
import Foreign.Ptr

import TensorDouble
import TensorDoubleMath (sigmoid, (!*), addmv)
import TensorDoubleRandom
import TensorRaw
import Random
import TensorTypes
import TensorUtils

data Weights = W {
  biases :: TensorDouble_,
  nodes :: TensorDouble_
  } deriving (Eq, Show)

-- instance Show Weights where
--   show w = "TODO implementat show"

data Network :: * where
  O :: Weights -> Network
  (:~) :: Weights -> Network -> Network
  deriving Show

infixr 5 :~

dispWeights w = do
  putStrLn "Biases:"
  disp_ (biases w)
  putStrLn "Weights:"
  disp_ (nodes w)

randomWeights i o = do
  gen <- newRNG
  let w1 = W { biases = tensorNew_ (D1 o), nodes = tensorNew_ (D2 i o) }
  dispWeights w1
  b <- uniformT (biases w1) gen (-1.0) (1.0)
  w <- uniformT (nodes w1) gen (-1.0) (1.0)
  pure W { biases = b, nodes = w }

randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> TensorDouble_ -> TensorDouble_
runLayer (W wB wN) v = addmv 1.0 wB 1.0 wN v

runNet :: Network -> TensorDouble_ -> TensorDouble_
runNet (O w) v = sigmoid (runLayer w v)
runNet (w :~ n') v = let v' = sigmoid (runLayer w v) in runNet n' v'

main = do
  w <- randomWeights 5 3
  dispWeights w
  net <- randomNet 5 [3, 2, 4] 2
  putStrLn "Done"
