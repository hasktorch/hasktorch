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

module Torch.Typed.AuxSpec where

import           Prelude                 hiding ( sin )
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

checkDynamicTensorAttributes
  :: forall device dtype shape
   . (TensorOptions shape dtype device)
  => Tensor device dtype shape
  -> IO ()
checkDynamicTensorAttributes t = do
  D.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  D.dtype untyped `shouldBe` optionsRuntimeDType @shape @dtype @device
  D.shape untyped `shouldBe` optionsRuntimeShape @shape @dtype @device
  where untyped = toDynamic t

allFloatingPointDTypes :: _
allFloatingPointDTypes = withHalf standardFloatingPointDTypes

standardFloatingPointDTypes :: _
standardFloatingPointDTypes = Proxy @ 'D.Float :. Proxy @ 'D.Double :. HNil

allDTypes :: _
allDTypes = withHalf almostAllDTypes

withHalf :: _ -> _
withHalf dtypes = Proxy @ 'D.Half :. dtypes

almostAllDTypes :: _
almostAllDTypes = withBool standardDTypes

withBool :: _ -> _
withBool dtypes = Proxy @ 'D.Bool :. dtypes

standardDTypes :: _
standardDTypes =
  Proxy @ 'D.UInt8
    :. Proxy @ 'D.Int8
    :. Proxy @ 'D.Int16
    :. Proxy @ 'D.Int32
    :. Proxy @ 'D.Int64
    :. standardFloatingPointDTypes

justCPU :: _
justCPU = Proxy @'( 'D.CPU, 0) :. HNil

justCUDA :: forall deviceIndex . _
justCUDA = Proxy @'( 'D.CUDA, deviceIndex) :. HNil

spec :: Spec
spec = return ()
