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

type SN = StaticNetwork
type SW = StaticWeights

data LayerType = LTrivial | LLinear | LSigmoid | LRelu

{- State representations of layers -}

data StaticWeights (i :: Nat) (o :: Nat) = SW {
  biases :: TDS '[o],
  weights :: TDS '[o, i]
  } deriving (Show)

{- Layer representation in a network, wraps layer state as a type argument -}

data Layer (l :: LayerType) (i :: Nat) (o :: Nat) where
  LayerTrivial :: Layer 'LTrivial i i
  LayerLinear  :: (TDS '[o, i]) -> Layer 'LLinear i o
  LayerSigmoid :: Layer 'LSigmoid i i
  LayerRelu    :: Layer 'LRelu i i

type Gradient l i o = Layer l i o

type Sensitivity i = TDS '[i]

data Table :: Nat -> [Nat] -> Nat -> * where
  V :: (KnownNat i, KnownNat o) =>
       (Gradient l i o, Sensitivity i) -> Table i '[] o
  (:&~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          (Gradient l i o, Sensitivity i) -> Table h hs o -> Table i (h ': hs) o

-- updateTensor :: SingI d => TDS d -> TDS d -> Double -> TDS d
-- updateTensor t dEdt learningRate = t ^+^ (learningRate *^ dEdt)

updateMatrix :: (KnownNat o, KnownNat i) =>
  TDS [o, i] -> TDS [o, i] -> Double -> TDS [o, i]
updateMatrix t dEdt learningRate = t ^+^ (learningRate *^ dEdt)

updateLayer :: (KnownNat i, KnownNat o) =>
  Layer l i o -> Gradient l i o -> Layer l i o
updateLayer LayerTrivial _ = LayerTrivial
updateLayer LayerSigmoid _ = LayerSigmoid
updateLayer LayerRelu _ = LayerRelu
updateLayer (LayerLinear w) (LayerLinear gradient) = 
  LayerLinear (updateMatrix undefined undefined 0.0001)

forwardProp :: forall l i o . (KnownNat i, KnownNat o) =>
  TDS '[i] -> (Layer l i o) -> TDS '[o]
forwardProp t LayerTrivial = t
forwardProp t (LayerLinear w) =
  tds_resize ( w !*! t')
    where
      t' = (tds_resize t :: TDS '[i, 1])
forwardProp t LayerSigmoid = tds_sigmoid t
forwardProp t LayerRelu = (tds_gtTensorT t (tds_new)) ^*^ t
-- forwardProp t (LayerLinear (SW b w) :: Layer i o) =
--   tds_resize ( w !*! t' + b')
--   where
--     t' = (tds_resize t :: TDS '[i, 1])
--     b' = tds_resize b

backProp :: forall l i o . (KnownNat i, KnownNat o) =>
    Sensitivity o -> (Layer l i o) -> (Gradient l i o, Sensitivity i)
backProp dEds layer = (undefined, undefined)

trivial' :: SingI d => TDS d -> TDS d
trivial' t = tds_init 1.0

sigmoid' :: SingI d => TDS d -> TDS d
sigmoid' t = (tds_sigmoid t) ^*^ ((tds_init 1.0) ^-^ tds_sigmoid t)

relu' :: SingI d => TDS d -> TDS d
relu' t = (tds_gtTensorT t (tds_new))

forwardNetwork :: forall i h o . TDS '[i] -> SN i h o  -> TDS '[o]
forwardNetwork t (O w) = forwardProp t w
forwardNetwork t (h :~ n) = forwardNetwork (forwardProp t h) n

mkW :: (SingI i, SingI o) => SW i o
mkW = SW b n
  where (b, n) = (tds_new, tds_new)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       Layer l i o -> SN i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          Layer l i h -> SN h hs o -> SN i (h ': hs) o

-- set precedence to chain layers without adding parentheses
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

dispN :: SN h hs c -> IO ()
dispN (O w) = dispL w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN n'

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
  tds_p $ tds_gtTensorT t tds_new

  dispN net
  tds_p $ forwardNetwork (tds_init 5.0) net

  putStrLn "Done"
