{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Typed leaves for "Torch.StateDict": loading a checkpoint tensor into a
-- @Tensor device dtype shape@ (or @Parameter@, or @NamedTensor@) checks the
-- stored shape and dtype against the static ones and moves the data to the
-- static device.  A mismatched checkpoint therefore fails at the load site,
-- with the offending path and both shapes in the error — before anything
-- runs.
--
-- Sized vectors load as numbered children (@prefix.0@ …), like lists in the
-- untyped module, but with the length known statically a missing index is an
-- error rather than the end of the list.
module Torch.Typed.StateDict () where

import Control.Monad (forM, when)
import Data.Maybe (fromJust)
import Data.Proxy (Proxy (..))
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import GHC.TypeLits (KnownNat, natVal)
import qualified Torch.DType as D
import Torch.StateDict
import qualified Torch.Tensor as D
import Torch.Typed.Parameter (Parameter, makeIndependent, toDependent)
import Torch.Typed.Tensor

instance
  TensorOptions shape dtype device =>
  FromStateDict (Tensor device dtype shape)
  where
  fromStateDict sd path = do
    t <- fromStateDict sd path
    let expectedShape = optionsRuntimeShape @shape @dtype @device
        expectedDType = optionsRuntimeDType @shape @dtype @device
    when (D.shape t /= expectedShape) $
      fail $
        "state dict key "
          <> show path
          <> " has shape "
          <> show (D.shape t)
          <> ", but the model expects "
          <> show expectedShape
    when (D.dtype t /= expectedDType) $
      fail $
        "state dict key "
          <> show path
          <> " has dtype "
          <> show (D.dtype t)
          <> ", but the model expects "
          <> show expectedDType
    pure (UnsafeMkTensor (D.toDevice (optionsRuntimeDevice @shape @dtype @device) t))

instance ToStateDict (Tensor device dtype shape) where
  toStateDict t path = toStateDict (toDynamic t) path

instance
  TensorOptions shape dtype device =>
  FromStateDict (Parameter device dtype shape)
  where
  fromStateDict sd path = makeIndependent =<< fromStateDict sd path

instance ToStateDict (Parameter device dtype shape) where
  toStateDict p path = toStateDict (toDependent p) path

instance
  (TensorOptions (ToNats shape) dtype device) =>
  FromStateDict (NamedTensor device dtype shape)
  where
  fromStateDict sd path = fromUnnamed <$> fromStateDict @(Tensor device dtype (ToNats shape)) sd path

instance ToStateDict (NamedTensor device dtype shape) where
  toStateDict t path = toStateDict (toDynamic t) path

instance
  (KnownNat n, FromStateDict a) =>
  FromStateDict (Vector n a)
  where
  fromStateDict sd path = do
    let n = fromIntegral (natVal (Proxy @n)) :: Int
    xs <- forM [0 .. n - 1] $ \i -> fromStateDict sd (childPath path (show i))
    pure (fromJust (V.fromList xs))

instance ToStateDict a => ToStateDict (Vector n a) where
  toStateDict v path =
    toStateDict (V.toList v) path
