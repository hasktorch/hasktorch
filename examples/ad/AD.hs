{-# LANGUAGE DataKinds, GADTs, KindSignatures, RankNTypes, TypeFamilies, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables                                           #-}
{-# LANGUAGE LambdaCase                                                    #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-imports #-}

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

dispL :: forall o i . (KnownNat o, KnownNat i) => Layer i o -> IO ()
dispL _ = do
    let inVal = natVal (Proxy :: Proxy i)
    let outVal = natVal (Proxy :: Proxy o)
    print $ "inputs: " ++ (show inVal) ++ "    outputs: " ++ show (outVal)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispL w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN n'

type ForwardFunction = forall (i :: Nat) (o :: Nat) . TDS '[i] -> TDS '[o]

foo :: Sigmoid 3 3
foo = Sigmoid

l1 :: Layer 10 7
l1 = linear
l2 :: Layer 7 7
l2 = sigmoid
l3 :: Layer 7 4
l3 = linear
l4 :: Layer 4 4
l4 = sigmoid
lo :: Layer 4 2
lo = linear

-- hh :: Layer 7 7
-- hh = sigmoid
-- ho :: Layer 4 2
-- ho = mkW

net :: SN 4 '[] 2
net = O lo

net2 :: SN 4 '[4] 2
net2 = l4 :~ O lo

main :: IO ()
main = do
  print s
  dispN net2
  putStrLn "Done"
  where
    s = Sigmoid :: Sigmoid (3 :: Nat) (4 :: Nat)
