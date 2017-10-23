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

import StaticTensorDouble
import TensorDouble
import TensorDoubleMath (sigmoid, (!*), addmv)
import TensorDoubleRandom
import Random
import TensorTypes
import TensorUtils

{- Statically Typed Implementation -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights i o = SW {
  s_biases :: TDS 1 '[o],
  s_nodes :: TDS 2 '[i, o]
  } deriving (Show)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O_s :: SW i o -> StaticNetwork i '[] o
  (:&~) :: (KnownNat h) => SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :&~

main = do
    putStrLn "Done"
