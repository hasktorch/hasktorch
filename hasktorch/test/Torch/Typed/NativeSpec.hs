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
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data UnarySpec =
    AbsSpec
  | SignSpec
  | FracSpec
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
  | PowSpec 
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

instance (TensorOptions shape dtype device)
  => Apply
       UnarySpec
       (Proxy '(device, dtype, shape))
       (() -> IO ())
 where
  apply SinSpec _ _ = do
    let t = sin zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t


spec :: Spec
spec = do
  describe "unary native ops" $ do
    it "sin" (hfoldrM @IO SinSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "maxPool2d" $ do
      let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
      checkDynamicTensorAttributes c
  describe "binary native ops" $ return ()
