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
import TensorDoubleMath (sigmoid)
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

runLayer w wB wN v = undefined

-- runNet (O w) v = undefined
-- runNet w :~ n' = undefined

main = do
  w <- randomWeights 5 3
  dispWeights w
  randomNet 5 [3, 2, 4] 2
  putStrLn "Done"
