{-# LANGUAGE DataKinds, GADTs, KindSignatures,  TypeFamilies, TypeOperators #-}
{-# LANGUAGE LambdaCase                                                     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes                                                     #-}
{-# LANGUAGE ScopedTypeVariables                                            #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-imports -Wno-missing-signatures #-}

module Main where

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights (i :: Nat) (o :: Nat) = SW {
  biases :: TDS '[o],
  nodes :: TDS '[o, i]
  } deriving (Show)

data Sigmoid (i :: Nat) (o :: Nat) =
  Sigmoid deriving Show

data Layer (i :: Nat) (o :: Nat) =
  LinearLayer (SW i o)
  | SigmoidLayer (Sigmoid i o)
  deriving Show

type ForwardFunction = forall (i :: Nat) (o :: Nat) . TDS '[i] -> TDS '[o]

class Prop l (i :: Nat) (o :: Nat) where
  forwardProp :: TDS '[i] -> l i o -> TDS '[o]

-- instance (KnownNat i, KnownNat o) => Prop (Layer i o) where
--   forwardProp inTensor LinearLayer (SW i o) =
--     undefined

mkW :: (SingI i, SingI o) => SW i o
mkW = SW b n
  where (b, n) = (tds_new, tds_new)

sigmoid :: forall d . (SingI d) => Layer d d
sigmoid = SigmoidLayer (Sigmoid :: Sigmoid d d)

linear  :: forall i o . (SingI i, SingI o) => Layer i o
linear = LinearLayer (mkW :: SW i o)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       Layer i o -> SN i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          Layer i h -> SN h hs o -> SN i (h ': hs) o

-- set precedence to chain layers without adding parentheses
infixr 5 :~

dispL :: forall o i . (KnownNat o, KnownNat i) => Layer i o -> IO ()
dispL layer = do
    let inVal = natVal (Proxy :: Proxy i)
    let outVal = natVal (Proxy :: Proxy o)
    print layer
    print $ "inputs: " ++ (show inVal) ++ "    outputs: " ++ show (outVal)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispL w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN n'

li :: Layer 10 7
li = linear

l2 :: Layer 7 7
l2 = sigmoid

l3 :: Layer 7 4
l3 = linear

l4 :: Layer 4 4
l4 = sigmoid

lo :: Layer 4 2
lo = linear

net2 = li :~ l2 :~ l3 :~ l4 :~ O lo

-- net2 :: SN 10 '[7, 7, 4, 4, 2] 2
-- n4et2 = l1 :~ l2 :~ l3 :~ l4 :~ O lo

main :: IO ()
main = do
  print s
  dispN net2
  putStrLn "Done"
  where
    s = Sigmoid :: Sigmoid (3 :: Nat) (4 :: Nat)
