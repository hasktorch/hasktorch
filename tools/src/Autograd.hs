{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}

module Main where

-- experimental AD implementation
-- see Just Le's writeup
-- https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html

import Control.Exception.Base (assert)
import Data.Monoid ((<>))
import Data.Maybe (fromJust)
import Foreign.C.Types
import Foreign.Ptr

import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom
import THRandom
import THStorage
import THTypes
import TorchTensor

data Weights = W {
  weights :: TensorDouble
  } deriving (Eq)

instance Show Weights where
  show w = "TODO implementat show"

data Network :: * where
  O :: Weights -> Network
  (:&~) :: Weights -> Network -> Network

infixr 5 :&~

randInit sz lower upper = do
  gen <- c_THGenerator_new
  t <- fromJust $ tensorNew sz
  mapM_ (\x -> do
            c_THDoubleTensor_uniform t gen lower upper
            disp t
        ) [0..3]

showWeights w label = do
  putStrLn label
  disp w

test = do
  gen <- c_THGenerator_new
  w1 <- fromJust $ tensorNew [5]
  c_THDoubleTensor_uniform w1 gen (-1.0) (1.0)
  showWeights w1 "w1"
  -- invlogit 
  --w2 <- fromJust $ tensorNew [5]

-- runLayer :: Weights -> Vector Double -> Vector Double
-- runLayer (W w) v = c_THDoubleTensor_dot w v

-- runNet :: Network -> Vector Double -> Vector Double
-- runNet (O w)      !v = logistic (runLayer w v)
-- runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
--                        in  runNet n' v'

mvTest = do
  mat <- fromJust $ tensorNew [5,3]
  vec <- fromJust $ tensorNew [3]
  res <- fromJust $ tensorNew [5]
  zero <- fromJust $ tensorNew [5]
  print $ "dimension check matrix:" <>
    show (c_THDoubleTensor_nDimension mat == 2)
  print $ "dimension check vector:" <>
    show (c_THDoubleTensor_nDimension vec == 1)
  c_THDoubleTensor_fill mat 3.0
  c_THDoubleTensor_fill vec 2.0
  disp mat
  disp vec
  c_THDoubleTensor_addmv res 1.0 zero 1.0 mat vec
  disp res

main = do
  gen <- c_THGenerator_new
  w1 <- fromJust $ tensorNew [5]
  w2 <- fromJust $ tensorNew [3]
  w3 <- fromJust $ tensorNew [5]
  c_THDoubleTensor_uniform w1 gen (-1.0) (1.0)
  showWeights w1 "w1"
  c_THDoubleTensor_uniform w2 gen (-1.0) (1.0)
  showWeights w2 "w2"
  c_THDoubleTensor_uniform w3 gen (-1.0) (1.0)
  showWeights w3 "w3"
  let (ih, hh, ho) = (W w1, W w2, W w3)
  let network = ih :&~ hh :&~ O ho
  putStrLn "Done"
  pure ()
