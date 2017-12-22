{-# LANGUAGE DataKinds, GADTs, TypeFamilies, TypeOperators    #-}
{-# LANGUAGE LambdaCase                                                        #-}
{-# LANGUAGE MultiParamTypeClasses                                             #-}
{-# LANGUAGE ScopedTypeVariables                                               #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-imports -Wno-missing-signatures -Wno-unused-matches #-}

module Main where

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleMath
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

type SN = StaticNetwork
type SN2 = StaticNetwork2
type SW = StaticWeights

data StaticWeights (i :: Nat) (o :: Nat) = SW {
  biases :: TDS '[o],
  weights :: TDS '[o, i]
  } deriving (Show)

data Sigmoid (i :: Nat) =
  Sigmoid deriving Show

data Relu (i :: Nat) =
  Relu deriving Show

data Layer (i :: Nat) (o :: Nat) where
  LinearLayer  :: SW i o    -> Layer i o
  SigmoidLayer :: Sigmoid i -> Layer i i
  ReluLayer :: Relu i -> Layer i i

instance (KnownNat i, KnownNat o) => Show (Layer i o) where
  show (LinearLayer x) = "LinearLayer "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (SigmoidLayer x) = "SigmoidLayer "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (ReluLayer x) = "ReluLayer "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))


forwardProp :: forall i o . (KnownNat i, KnownNat o) =>
  TDS '[i] -> (Layer i o) -> TDS '[o]

forwardProp t (LinearLayer (SW b w) :: Layer i o) =
  tds_resize ( w !*! t' + b')
  where
    t' = (tds_resize t :: TDS '[i, 1])
    b' = tds_resize b

forwardProp t (SigmoidLayer Sigmoid) =
  tds_sigmoid t

forwardProp t (ReluLayer Relu) =
  undefined -- TODO

forwardNetwork :: forall i h o . TDS '[i] -> SN i h o  -> TDS '[o]
forwardNetwork t (O w) = forwardProp t w
forwardNetwork t (h :~ n) = forwardNetwork (forwardProp t h) n

mkW :: (SingI i, SingI o) => SW i o
mkW = SW b n
  where (b, n) = (tds_new, tds_new)

sigmoid :: forall d . (SingI d) => Layer d d
sigmoid = SigmoidLayer (Sigmoid :: Sigmoid d)

relu :: forall d . (SingI d) => Layer d d
relu = ReluLayer (Relu :: Relu d)

linear  :: forall i o . (SingI i, SingI o) => Layer i o
linear = LinearLayer (mkW :: SW i o)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       Layer i o -> SN i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          Layer i h -> SN h hs o -> SN i (h ': hs) o

data StaticNetwork2 :: [Nat] -> * where
  O2 :: (KnownNat i, KnownNat o) =>
       Layer i o -> SN2 '[i, o]
  (:&~) :: (KnownNat h, KnownNat i) =>
          Layer i h -> SN2 (h : hs) -> SN2 (i ': h ': hs)

-- set precedence to chain layers without adding parentheses
infixr 5 :~
infixr 5 :&~

dispL :: forall o i . (KnownNat o, KnownNat i) => Layer i o -> IO ()
dispL layer = do
    let inVal = natVal (Proxy :: Proxy i)
    let outVal = natVal (Proxy :: Proxy o)
    print layer
    print $ "inputs: " ++ (show inVal) ++ "    outputs: " ++ show (outVal)

dispN :: SN h hs c -> IO ()
dispN (O w) = dispL w
dispN (w :~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN n'

dispN2 :: SN2 (h:hs) -> IO ()
dispN2 (O2 w) = dispL w
dispN2 (w :&~ n') = putStrLn "\nCurrent Layer ::::" >> dispL w >> dispN2 n'

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

net = li :~ l2 :~ l3 :~ l4 :~ O lo
net2 = li :&~ l2 :&~ l3 :&~ l4 :&~ O2 lo

fstLayer ::
  forall i h hs o . SN i (h : hs) o -> Layer i h
fstLayer (f :~ r) = f

fstLayer2 ::
  forall i h hs . SN2 (i : h : hs) -> Layer i h
fstLayer2 (O2 o) = o
fstLayer2 (f :&~ r) = f

rstLayer2 ::
  forall i h hs . SN2 (i : h : hs) -> SN2 (h : hs)
rstLayer2 (O2 o) = undefined
rstLayer2 (f :&~ r) = r

main :: IO ()
main = do

  gen <- newRNG
  t <- tds_normal gen 0.0 5.0 :: IO (TDS '[10])
  tds_p $ tds_gtTensorT t tds_new

  print s
  dispN net
  tds_p $ forwardNetwork (tds_init 5.0) net
  dispN2 net2
  dispL $ fstLayer2 . rstLayer2 . rstLayer2 . rstLayer2 . rstLayer2  $ net2


  putStrLn "Done"
  where
    s = Sigmoid :: Sigmoid (3 :: Nat)
