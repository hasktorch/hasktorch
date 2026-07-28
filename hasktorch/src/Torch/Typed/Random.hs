{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Random where

import Data.Word (Word64)
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Random as D
import Torch.Typed.Auxiliary
import Torch.Typed.Tensor

newtype PureGenerator (device :: (D.DeviceType, Nat)) = UnsafePureGenerator D.PureGenerator

mkPureGenerator :: forall device. KnownDevice device => Word64 -> IO (PureGenerator device)
mkPureGenerator seed = UnsafePureGenerator <$> D.mkPureGenerator (deviceVal @device) seed

-- | The shape multinomial sampling produces: the trailing dimension is
-- replaced by the number of samples drawn, the leading batch dimensions are
-- carried through.
type family MultinomialShape (samples :: Nat) (shape :: [Nat]) :: [Nat] where
  MultinomialShape samples '[n] = '[samples]
  MultinomialShape samples (m ': ns) = m ': MultinomialShape samples ns

-- | Sample indices from the categorical distributions along the last dimension
-- of the input. ATen draws only from floating point probabilities.
multinomial ::
  forall samples shape dtype device.
  ( KnownNat samples,
    DTypeIsFloatingPoint device dtype
  ) =>
  -- | replacement
  Bool ->
  -- | input
  Tensor device dtype shape ->
  -- | generator
  PureGenerator device ->
  -- | output
  (Tensor device 'D.Int64 (MultinomialShape samples shape), PureGenerator device)
multinomial replacement input (UnsafePureGenerator g) =
  let (out, g') = D.multinomial (toDynamic input) (natValI @samples) replacement g
   in (UnsafeMkTensor out, UnsafePureGenerator g')
