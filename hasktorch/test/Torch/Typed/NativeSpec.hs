{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NativeSpec
  ( Torch.Typed.NativeSpec.spec
  )
where

import           Prelude                 hiding ( all
                                                , any
                                                , sin
                                                , sinh
                                                , cos
                                                , cosh
                                                , tan
                                                , tanh
                                                , asin
                                                , asinh
                                                , acos
                                                , acosh
                                                , atan
                                                , atanh
                                                , abs
                                                , max
                                                , min
                                                , exp
                                                , log
                                                , round
                                                , floor
                                                )
import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Proxy
import           Data.Reflection

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data UnaryAllDTypesSpec =
    SignSpec

data UnaryStandardDTypesSpec =
    AbsSpec

data UnaryFloatingPointDTypesSpec =
    FracSpec
  | CeilSpec
  | FloorSpec
  | RoundSpec
  | TruncSpec
  | ErfSpec
  | ErfcSpec
  | ErfinvSpec
  | ExpSpec
  | Expm1Spec
  | LogSpec
  | Log1pSpec
  | Log2Spec
  | Log10Spec
  | LgammaSpec
  | DigammaSpec
  | ReluSpec
  | SeluSpec
  | GeluSpec
  | SigmoidSpec
  | SinSpec
  | SinhSpec
  | CosSpec
  | AcosSpec
  | TanSpec
  | TanhSpec
  | SqrtSpec
  | RsqrtSpec
  | OnesLikeSpec
  | ZerosLikeSpec
  | RandLikeSpec
  | RandnLikeSpec

instance ( TensorOptions shape dtype device
         , DTypeIsNotHalf dtype
         , DTypeIsNotBool dtype
         )
  => Apply
       UnaryStandardDTypesSpec
       (Proxy '(device, dtype, shape))
       (() -> IO ())
 where
  apply AbsSpec _ _ = do
    let t = abs ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t

instance (TensorOptions shape dtype device)
  => Apply
       UnaryAllDTypesSpec
       (Proxy '(device, dtype, shape))
       (() -> IO ())
 where
  apply SignSpec _ _ = do
    let t = sign ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t

instance ( TensorOptions shape dtype device
         , IsFloatingPoint dtype
         , DTypeIsNotHalf dtype
         )
  => Apply
       UnaryFloatingPointDTypesSpec
       (Proxy '(device, dtype, shape))
       (() -> IO ())
 where
  apply FracSpec _ _ = do
    let t = frac ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply CeilSpec _ _ = do
    let t = ceil ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply FloorSpec _ _ = do
    let t = floor ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply TruncSpec _ _ = do
    let t = trunc ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfSpec _ _ = do
    let t = erf ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfcSpec _ _ = do
    let t = erfc ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfinvSpec _ _ = do
    let t = erfinv ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ExpSpec _ _ = do
    let t = exp ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Expm1Spec _ _ = do
    let t = expm1 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply LogSpec _ _ = do
    let t = log ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log1pSpec _ _ = do
    let t = log1p ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log2Spec _ _ = do
    let t = log2 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log10Spec _ _ = do
    let t = log10 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply LgammaSpec _ _ = do
    let t = lgamma ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply DigammaSpec _ _ = do
    let t = digamma ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SinSpec _ _ = do
    let t = sin ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t


spec :: Spec
spec = do
  describe "unary native ops" $ do
    it "abs" (hfoldrM @IO AbsSpec () (standardDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "sign" (hfoldrM @IO SignSpec () (allDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "frac" (hfoldrM @IO FracSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "ceil" (hfoldrM @IO CeilSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "floor" (hfoldrM @IO FloorSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "trunc" (hfoldrM @IO TruncSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "erf" (hfoldrM @IO ErfSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "erfc" (hfoldrM @IO ErfcSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "erfinv" (hfoldrM @IO ErfinvSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "exp" (hfoldrM @IO ExpSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "expm1" (hfoldrM @IO Expm1Spec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "log" (hfoldrM @IO LogSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "log1p" (hfoldrM @IO Log1pSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "log2" (hfoldrM @IO Log2Spec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "log10" (hfoldrM @IO Log10Spec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "lgamma" (hfoldrM @IO LgammaSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "digamma" (hfoldrM @IO DigammaSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))

    it "sin" (hfoldrM @IO SinSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "maxPool2d" $ do
      let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
      checkDynamicTensorAttributes c
  describe "binary native ops" $ return ()
