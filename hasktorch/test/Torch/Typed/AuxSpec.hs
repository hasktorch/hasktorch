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
  D.dtype  untyped `shouldBe` optionsRuntimeDType  @shape @dtype @device
  D.shape  untyped `shouldBe` optionsRuntimeShape  @shape @dtype @device
 where untyped = toDynamic t

allFloatingPointDTypes :: forall device shape . _
allFloatingPointDTypes =
  withHalf @device @shape (standardFloatingPointDTypes @device @shape)

standardFloatingPointDTypes :: forall device shape . _
standardFloatingPointDTypes =
  Proxy @'(device, 'D.Float, shape)
    :. Proxy @'(device, 'D.Double, shape)
    :. HNil

allDTypes :: forall device shape . _
allDTypes = withHalf @device @shape $ almostAllDTypes @device @shape

withHalf :: forall device shape . _ -> _
withHalf dtypes = Proxy @'(device, 'D.Half, shape) :. dtypes

almostAllDTypes :: forall device shape . _
almostAllDTypes = withHalf @device @shape $ standardDTypes @device @shape

withBool :: forall device shape . _ -> _
withBool dtypes = Proxy @'(device, 'D.Bool, shape) :. dtypes

standardDTypes :: forall device shape . _
standardDTypes =
  Proxy @'(device, 'D.UInt8, shape)
    :. Proxy @'(device, 'D.Int8, shape)
    :. Proxy @'(device, 'D.Int16, shape)
    :. Proxy @'(device, 'D.Int32, shape)
    :. Proxy @'(device, 'D.Int64, shape)
    :. standardFloatingPointDTypes @device @shape

spec :: Spec
spec = return ()
