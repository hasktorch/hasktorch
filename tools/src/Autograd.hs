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
import TensorDoubleMath
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

-- randomWeights :: MonadRandom m => Int -> Int -> m Weights
-- randomWeights i o = do
--   let gen = newRNG
--       wB = uniform gen (-1.0) 1.0
--         let wB = randomVector  seed1 Uniform o * 2 - 1
--         wN = uniformSample seed2 o (replicate i (-1, 1))
--     return $ W wB wN


data Network :: * where
  O :: Weights -> Network
  (:&~) :: Weights -> Network -> Network

infixr 5 :&~

dispWeights w = do
  disp_ (biases w)
  disp_ (nodes w)

test = do
  gen <- newRNG
  let w1 = W { biases = tensorNew_ (D1 3), nodes = tensorNew_ (D2 3 2) }
  dispWeights w1
  b <- uniformT (biases w1) gen (-1.0) (1.0)
  disp_ b
  pure ()

main = do
  test
  putStrLn "Done"
