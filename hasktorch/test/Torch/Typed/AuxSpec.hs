{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.AuxSpec where

import Control.Exception.Safe
import Data.Proxy
import Data.Reflection
import Foreign.Storable
import GHC.TypeLits
import System.IO.Unsafe
import Test.Hspec
import Test.QuickCheck
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor
import Prelude hiding (sin)

instance Semigroup Spec where
  (<>) a b = a >> b

instance Monoid Spec where
  mempty = pure ()

checkDynamicTensorAttributes ::
  forall device dtype shape.
  (TensorOptions shape dtype device) =>
  Tensor device dtype shape ->
  IO ()
checkDynamicTensorAttributes t = do
  D.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  D.dtype untyped `shouldBe` optionsRuntimeDType @shape @dtype @device
  D.shape untyped `shouldBe` optionsRuntimeShape @shape @dtype @device
  where
    untyped = toDynamic t

allFloatingPointDTypes :: _
allFloatingPointDTypes = withHalf standardFloatingPointDTypes

standardFloatingPointDTypes :: _
standardFloatingPointDTypes = Proxy @'D.Float :. Proxy @'D.Double :. HNil

allDTypes :: _
allDTypes = withHalf almostAllDTypes

withHalf :: _ -> _
withHalf dtypes = Proxy @'D.Half :. dtypes

almostAllDTypes :: _
almostAllDTypes = withBool standardDTypes

withBool :: _ -> _
withBool dtypes = Proxy @'D.Bool :. dtypes

standardDTypes :: _
standardDTypes =
  Proxy @'D.UInt8
    :. Proxy @'D.Int8
    :. Proxy @'D.Int16
    :. Proxy @'D.Int32
    :. Proxy @'D.Int64
    :. standardFloatingPointDTypes

cpu :: _
cpu = Proxy @'( 'D.CPU, 0)

cuda0 :: _
cuda0 = Proxy @'( 'D.CUDA, 0)

availableDevices =
  let hasCuda = unsafePerformIO $ cast0 ATen.hasCUDA
   in [D.Device {D.deviceType = D.CPU, D.deviceIndex = 0}]
        <> ( if hasCuda
               then [D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0}]
               else mempty
           )

spec :: Spec
spec = return ()
