{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.AuxiliarySpec where

import Data.Proxy
import System.IO.Unsafe
import Test.Hspec (Spec, shouldBe)
import Test.QuickCheck ()
import qualified Torch as Torch (device, dtype, shape)
import Torch.Internal.Cast (cast0)
import Torch.Internal.Managed.Type.Context (hasCUDA)
import Torch.Typed

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
  Torch.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  Torch.dtype untyped `shouldBe` optionsRuntimeDType @shape @dtype @device
  Torch.shape untyped `shouldBe` optionsRuntimeShape @shape @dtype @device
  where
    untyped = toDynamic t

allFloatingPointDTypes :: _
allFloatingPointDTypes = withHalf standardFloatingPointDTypes

standardFloatingPointDTypes :: _
standardFloatingPointDTypes = Proxy @'Float :. Proxy @'Double :. HNil

allDTypes :: _
allDTypes = withHalf almostAllDTypes

withHalf :: _ -> _
withHalf dtypes = Proxy @'Half :. dtypes

almostAllDTypes :: _
almostAllDTypes = withBool standardDTypes

withBool :: _ -> _
withBool dtypes = Proxy @'Bool :. dtypes

standardDTypes :: _
standardDTypes =
  Proxy @'UInt8
    :. Proxy @'Int8
    :. Proxy @'Int16
    :. Proxy @'Int32
    :. Proxy @'Int64
    :. standardFloatingPointDTypes

cpu :: _
cpu = Proxy @'( 'CPU, 0)

cuda0 :: _
cuda0 = Proxy @'( 'CUDA, 0)

availableDevices :: [Device]
availableDevices =
  let hasCuda = unsafePerformIO $ cast0 hasCUDA
   in [Device {deviceType = CPU, deviceIndex = 0}]
        <> ( if hasCuda
               then [Device {deviceType = CUDA, deviceIndex = 0}]
               else mempty
           )

spec :: Spec
spec = return ()
