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

-- import THDoubleTensor
-- import THDoubleTensorMath
-- import THDoubleTensorRandom
-- import THRandom
-- import THStorage
-- import THTypes

import Tensor
import TensorRaw
import TensorRandom
import TensorTypes

data Weights = W {
  weights :: TensorDouble_
  } deriving (Eq)

instance Show Weights where
  show w = "TODO implementat show"

data Network :: * where
  O :: Weights -> Network
  (:&~) :: Weights -> Network -> Network

infixr 5 :&~

test = do
  gen <- newRNG
  let w1 = tensorNew_ (D1 5)
  pure ()

-- runLayer :: Weights -> Vector Double -> Vector Double
-- runLayer (W w) v = c_THDoubleTensor_dot w v

-- runNet :: Network -> Vector Double -> Vector Double
-- runNet (O w)      !v = logistic (runLayer w v)
-- runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
--                        in  runNet n' v'

{-
simpler but unsafe version
-}

main = do
  putStrLn "Done"
