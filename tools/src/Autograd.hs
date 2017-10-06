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
import TensorRaw
import Random
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

main = do
  putStrLn "Done"
