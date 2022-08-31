{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Torch.Typed.VLTensor where

import Data.Proxy
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as Untyped
import qualified Torch.Functional.Internal as Internal
import qualified Torch.Tensor as Untyped
import Torch.Typed.Auxiliary
import Torch.Typed.Tensor
import Unsafe.Coerce (unsafeCoerce)

-- | A variable length tensor. The length cannot be determined in advance.
data VLTensor (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (shape :: [Nat]) = forall n. KnownNat n => VLTensor (Tensor device dtype (n : shape))

instance Show (VLTensor device dtype shape) where
  show input =
    case input of
      VLTensor v -> show v

fromVLTensor ::
  forall n device dtype shape.
  ( KnownNat n,
    TensorOptions shape dtype device,
    KnownShape shape
  ) =>
  VLTensor device dtype shape ->
  Maybe (Tensor device dtype (n : shape))
fromVLTensor (VLTensor input) =
  if shape input == shapeVal @(n : shape)
    then Just (unsafeCoerce input)
    else Nothing

selectIndexes :: forall n device dtype shape. Tensor device dtype (n : shape) -> Tensor device 'D.Bool '[n] -> VLTensor device dtype shape
selectIndexes input boolTensor =
  let output = toDynamic input Untyped.! toDynamic boolTensor
   in withNat (head $ Untyped.shape output) $ \(Proxy :: Proxy b) ->
        VLTensor $ UnsafeMkTensor @device @dtype @(b : shape) output

pack :: forall device dtype shape. [Tensor device dtype shape] -> VLTensor device dtype shape
pack input =
  let output = Untyped.stack (Untyped.Dim 0) $ map toDynamic input
   in withNat (head $ Untyped.shape output) $ \(Proxy :: Proxy n) ->
        VLTensor $ UnsafeMkTensor @device @dtype @(n : shape) output

unpack :: forall device dtype shape. VLTensor device dtype shape -> [Tensor device dtype shape]
unpack input =
  case input of
    (VLTensor (input' :: Tensor device dtype (n : shape))) ->
      let output = Internal.unbind (toDynamic input') 0
       in map (UnsafeMkTensor @device @dtype @shape) output

