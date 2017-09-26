{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}

module Main where

-- experimental AD implementation
-- see Just Le's writeup
-- https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html

import Data.Maybe
import Foreign.C.Types
import Foreign.Ptr

import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom
import THRandom
import THTypes
import TorchTensor

data Weights = W {
  weights :: Ptr CTHDoubleTensor
  }

data Network :: * where
  O :: Weights -> Network
  (:&~) :: Weights -> Network -> Network

infixr 5 :&~

randInit = do
  gen <- c_THGenerator_new
  t <- fromJust $ tensorNew [5]
  putStrLn "initialized vector"
  disp t
  putStrLn "random vectors"
  mapM_ (\x -> do
            c_THDoubleTensor_uniform t gen (-1.0) (1.0)
            disp t
        ) [0..3]

showWeights w label = do
  putStrLn label
  disp w

main = do
  gen <- c_THGenerator_new
  w1 <- fromJust $ tensorNew [5]
  w2 <- fromJust $ tensorNew [5]
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
