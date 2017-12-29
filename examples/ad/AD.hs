{-# LANGUAGE DataKinds, GADTs, TypeFamilies, TypeOperators #-}
{-# LANGUAGE LambdaCase                                    #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses                         #-}
{-# LANGUAGE ScopedTypeVariables                           #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-imports -Wno-missing-signatures -Wno-unused-matches #-}

module Main where

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

data LayerType = LTrivial | LLinear | LSigmoid | LRelu

-- data AffineWeights (i :: Nat) (o :: Nat) = AW {
--   biases :: TDS '[o],
--   weights :: TDS '[o, i]
--   } deriving (Show)
-- type AW = AffineWeights

data Layer (l :: LayerType) (i :: Nat) (o :: Nat) where
  LayerTrivial :: Layer 'LTrivial i i
  LayerLinear  :: (TDS '[o, i]) -> Layer 'LLinear i o
  -- LayerAffine :: (AffineWeights i o) -> Layer 'LLinear i o
  LayerSigmoid :: Layer 'LSigmoid i i
  LayerRelu    :: Layer 'LRelu i i

type Gradient l i o = Layer l i o

type Sensitivity i = TDS '[i]

type Output o = TDS '[o]

data Table :: Nat -> [Nat] -> Nat -> * where
  T :: (KnownNat i, KnownNat o) => (Gradient l i o, Sensitivity i) -> Table i '[] o
  (:&~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          (Gradient l i o, Sensitivity i) -> Table h hs o -> Table i (h ': hs) o

data Values :: [Nat] -> Nat -> * where
  -- V :: KnownNat o => Output o -> Values '[] o
  -- VNil :: Values '[]
  V :: (KnownNat o) => Output o -> Values '[] o
  (:^~) :: (KnownNat h, KnownNat o) =>
           Output h -> Values hs o -> Values (h ': hs) o

data Network :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       Layer l i o -> NW i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          Layer l i h -> NW h hs o -> NW i (h ': hs) o

updateTensor :: SingI d => TDS d -> TDS d -> Double -> TDS d
updateTensor t dEdt learningRate = t ^+^ (learningRate *^ dEdt)

updateLayer :: (KnownNat i, KnownNat o) =>
  Double -> Layer l i o -> Gradient l i o -> Layer l i o
updateLayer _ LayerTrivial _ = LayerTrivial
updateLayer _ LayerSigmoid _ = LayerSigmoid
updateLayer _ LayerRelu _ = LayerRelu
updateLayer learningRate (LayerLinear w) (LayerLinear gradient) =
  LayerLinear (updateTensor w gradient learningRate)

-- TODO: write to Values
forwardProp :: forall l i o . (KnownNat i, KnownNat o) =>
  TDS '[i] -> (Layer l i o) -> TDS '[o]
forwardProp t LayerTrivial = t
forwardProp t (LayerLinear w) =
  tds_resize ( w !*! t')
    where
      t' = (tds_resize t :: TDS '[i, 1])
forwardProp t LayerSigmoid = tds_sigmoid t
forwardProp t LayerRelu = (tds_gtTensorT t (tds_new)) ^*^ t
-- forwardProp t (LayerAffine (AW b w) :: Layer i o) =
--   tds_resize ( w !*! t' + b')
--   where
--     t' = (tds_resize t :: TDS '[i, 1])
--     b' = tds_resize b

-- TODO: write to Table
backProp :: forall l i o . (KnownNat i, KnownNat o) =>
    Sensitivity o -> (Layer l i o) -> (Gradient l i o, Sensitivity i)
backProp dEds layer = (undefined, undefined)

trivial' :: SingI d => TDS d -> TDS d
trivial' t = tds_init 1.0

sigmoid' :: SingI d => TDS d -> TDS d
sigmoid' t = (tds_sigmoid t) ^*^ ((tds_init 1.0) ^-^ tds_sigmoid t)

relu' :: SingI d => TDS d -> TDS d
relu' t = (tds_gtTensorT t (tds_new))

-- forward prop, don't retain values
forwardNetwork :: forall i h o . TDS '[i] -> NW i h o  -> TDS '[o]
forwardNetwork t (O w) = forwardProp t w
forwardNetwork t (hh :~ hr) = forwardNetwork (forwardProp t hh) hr

-- forward prop, retain values
forwardNetwork' :: forall i h o . TDS '[i] -> NW i h o -> Values h o
-- forwardNetwork' t (h :~ O w) = V (forwardProp t h)
forwardNetwork' t (O olayer) = V (forwardProp t olayer)
forwardNetwork' t (h :~ hs) = output :^~ (forwardNetwork' output hs)
  where output = forwardProp t h

-- mkW :: (SingI i, SingI o) => AW i o
-- mkW = AW b n
--   where (b, n) = (tds_new, tds_new)

type NW = Network

infixr 5 :~

instance (KnownNat i, KnownNat o) => Show (Layer l i o) where
  show (LayerTrivial) = "LayerTrivial "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (LayerLinear x) = "LayerLinear "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (LayerSigmoid) = "LayerSigmoid "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (LayerRelu) = "LayerRelu "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))

dispL :: forall o i l . (KnownNat o, KnownNat i) => Layer l i o -> IO ()
dispL layer = do
    let inVal = natVal (Proxy :: Proxy i)
    let outVal = natVal (Proxy :: Proxy o)
    print layer
    print $ "inputs: " ++ (show inVal) ++ "    outputs: " ++ show (outVal)

dispN :: NW h hs c -> IO ()
dispN (O w) = putStrLn "\nOutput Layer ::::" >> dispL w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN n'

dispV :: Values hs o -> IO ()
dispV (V o)= putStrLn "\nOutput Layer ::::" >> tds_p o
dispV (v :^~ n) = putStrLn "\nCurrent Layer ::::" >> tds_p v >> dispV n

li :: Layer 'LLinear 10 7
li = LayerLinear tds_new
l2 :: Layer 'LSigmoid 7 7
l2 = LayerSigmoid
l3 :: Layer 'LLinear 7 4
l3 = LayerLinear tds_new
l4 :: Layer 'LSigmoid 4 4
l4 = LayerSigmoid
lo :: Layer 'LLinear 4 2
lo = LayerLinear tds_new

net = li :~ l2 :~ l3 :~ l4 :~ O lo

main :: IO ()
main = do

  gen <- newRNG
  t <- tds_normal gen 0.0 5.0 :: IO (TDS '[10])
  putStrLn "Input"
  tds_p $ tds_gtTensorT t tds_new

  putStrLn "Network"
  dispN net

  putStrLn "\nValues"
  let v = forwardNetwork' (tds_init 5.0) net
  dispV v

  putStrLn "Done"
