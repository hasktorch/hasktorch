{-# LANGUAGE DataKinds, GADTs, TypeFamilies, TypeOperators #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-imports -Wno-missing-signatures -Wno-unused-matches #-}
module Main where

import Torch
import qualified Torch.Core.Random as RNG
import System.IO.Unsafe (unsafePerformIO)

type Tensor = DoubleTensor

-- Promoted layer types promoted
data LayerType = LTrivial | LLinear | LAffine | LSigmoid | LRelu

data Layer (l :: LayerType) (i :: Nat) (o :: Nat) where
  LayerTrivial :: Layer 'LTrivial i i
  LayerLinear  :: (Tensor '[o, i]) -> Layer 'LLinear i o
  LayerAffine :: (AffineWeights i o) -> Layer 'LAffine i o
  LayerSigmoid :: Layer 'LSigmoid i i
  LayerRelu    :: Layer 'LRelu i i

type Gradient l i o = Layer l i o

type Sensitivity i = Tensor '[i]

type Output o = Tensor '[o]

-- Backprop sensitivity and parameter gradients
data Table :: Nat -> [Nat] -> Nat -> * where
  T :: (KnownNat i, KnownNat o) => (Gradient l i o, Sensitivity i) -> Table i '[] o
  (:&~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          (Gradient l i o, Sensitivity i) -> Table h hs o -> Table i (h ': hs) o

-- Forwardprop values
data Values :: [Nat] -> Nat -> * where
  V :: (KnownNat o) => Output o -> Values '[] o
  (:^~) :: (KnownNat h, KnownNat o) =>
           Output h -> Values hs o -> Values (h ': hs) o

-- Network architecture
data Network :: Nat -> [Nat] -> Nat -> * where
  O :: (KnownNat i, KnownNat o) =>
       Layer l i o -> NW i '[] o
  (:~) :: (KnownNat h, KnownNat i, KnownNat o) =>
          Layer l i h -> NW h hs o -> NW i (h ': hs) o

data AffineWeights (i :: Nat) (o :: Nat) = AW
  { biases :: Tensor '[o]
  , weights :: Tensor '[o, i]
  } deriving (Show)
type AW = AffineWeights

updateTensor :: SingDimensions d => Tensor d -> Tensor d -> Double -> Tensor d
updateTensor t dEdt learningRate = t ^+^ (learningRate *^ dEdt)

updateLayer :: (KnownNatDim i, KnownNatDim o) =>
  Double -> Layer l i o -> Gradient l i o -> Layer l i o
updateLayer _ LayerTrivial _ = LayerTrivial
updateLayer _ LayerSigmoid _ = LayerSigmoid
updateLayer _ LayerRelu _    = LayerRelu
updateLayer learningRate (LayerLinear w) (LayerLinear gradient) =
  LayerLinear (updateTensor w gradient learningRate)
updateLayer learningRate (LayerAffine w) (LayerAffine gradient) =
  LayerAffine $ AW (updateTensor (biases w) (biases gradient) learningRate)
                   (updateTensor (weights w) (weights gradient) learningRate)

forwardProp
  :: forall l i o . (KnownNatDim i, KnownNatDim o)
  => Tensor '[i] -> Layer l i o -> IO (Tensor '[o])
forwardProp t = \case
  LayerTrivial -> pure t
  LayerLinear w -> do
    t' :: Tensor '[i, 1] <- resizeAs t
    resizeAs (w !*! t')

  LayerSigmoid -> sigmoid t
  LayerRelu    -> constant 0 >>= gtTensorT t >>= \active -> pure $ active ^*^ t
  LayerAffine (AW b w) -> do
    t' :: Tensor '[i, 1] <- resizeAs t
    b' <- resizeAs b
    resizeAs ( w !*! t' ^+^ b')

-- TODO: write to Table
backProp :: forall l i o . (KnownNatDim i, KnownNatDim o) =>
    Sensitivity o -> (Layer l i o) -> (Gradient l i o, Sensitivity i)
backProp dEds LayerTrivial           = (LayerTrivial, dEds)
backProp dEds LayerSigmoid           = (LayerSigmoid, dEds ^*^ undefined)
backProp dEds LayerRelu              = (LayerRelu, dEds ^*^ undefined)
backProp dEds (LayerLinear w)        = (undefined , undefined)
backProp dEds (LayerAffine (AW w b)) = (undefined , undefined)

trivial' :: SingDimensions d => Tensor d -> IO (Tensor d)
trivial' t = constant 1

sigmoid' :: SingDimensions d => Tensor d -> IO (Tensor d)
sigmoid' t = do
  s <- sigmoid t
  o <- constant 1
  pure $ (s ^*^ o) ^-^ s

relu' :: SingDimensions d => Tensor d -> IO (Tensor d)
relu' t = constant 0 >>= gtTensorT t

-- forward prop, don't retain values
forwardNetwork :: forall i h o . (KnownDim i, KnownDim o) => Tensor '[i] -> NW i h o  -> IO (Tensor '[o])
forwardNetwork t = \case
  O w      -> forwardProp t w
  hh :~ hr -> forwardProp t hh >>= \p -> forwardNetwork p hr

-- forward prop, retain values
forwardNetwork' :: forall i h o . (KnownDim i, KnownDim o) => Tensor '[i] -> NW i h o -> IO (Values h o)
forwardNetwork' t = \case
   O olayer -> V <$> forwardProp t olayer
   h :~ hs  -> do
     output <- forwardProp t h
     adv <- forwardNetwork' output hs
     pure (output :^~ adv)

mkW :: (SingI i, KnownDim i, KnownDim o, SingI o) => IO (AW i o)
mkW = AW <$> new <*> new

type NW = Network

infixr 5 :~

instance (KnownNat i, KnownNat o) => Show (Layer l i o) where
  show (LayerTrivial) = "LayerTrivial "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))
  show (LayerLinear x) = "LayerLinear "
                         ++ (show (natVal (Proxy :: Proxy i))) ++ " "
                         ++ (show (natVal (Proxy :: Proxy o)))

  show (LayerAffine x) = "LayerAffine "
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

dispV :: KnownDim o => Values hs o -> IO ()
dispV (V o)= putStrLn "\nOutput Layer ::::" >> printTensor o
dispV (v :^~ n) = putStrLn "\nCurrent Layer ::::" >> printTensor v >> dispV n

li :: Layer 'LLinear 10 7
li = unsafePerformIO $ LayerLinear <$> new
{-# NOINLINE li #-}
l2 :: Layer 'LSigmoid 7 7
l2 = LayerSigmoid
l3 :: Layer 'LLinear 7 4
l3 = unsafePerformIO $ LayerLinear <$> new
{-# NOINLINE l3 #-}
l4 :: Layer 'LSigmoid 4 4
l4 = LayerSigmoid
lo :: Layer 'LLinear 4 2
lo = unsafePerformIO $ LayerLinear <$> new
{-# NOINLINE lo #-}

net = li :~ l2 :~ l3 :~ l4 :~ O lo

main :: IO ()
main = do
  gen <- RNG.new
  t :: Tensor '[10] <- normal gen 0 5

  putStrLn "Input"
  constant 0 >>= gtTensorT t >>= printTensor

  putStrLn "Network"
  dispN net

  putStrLn "\nValues"
  k  <- constant 5
  v <- forwardNetwork' k net
  dispV v

  putStrLn "Done"
