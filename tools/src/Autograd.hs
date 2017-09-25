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
import THTypes
import TorchTensor

data Weights = W {
  weights :: Ptr CTHDoubleTensor
  }

data Network :: * where
  O :: Weights -> Network
  (:&~) :: Weights -> Network -> Network

infixr 5 :&~

main = do
  w1 <- fromJust $ tensorNew [5]
  putStrLn "w1"
  disp w1
  w2 <- fromJust $ tensorNew [5]
  putStrLn "w2"
  disp w2
  putStrLn "w3"
  w3 <- fromJust $ tensorNew [5]
  disp w3
  ih <- W <$> (fromJust $ tensorNew [5])
  hh <- W <$> (fromJust $ tensorNew [5])
  ho <- W <$> (fromJust $ tensorNew [5])
  let network = ih :&~ hh :&~ O ho
  putStrLn "Done"
  pure ()
