{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | A bounded append buffer for sequences that grow at run time — the KV
-- cache shape of problem, where autoregressive decoding adds one position
-- per step and a static shape cannot follow.
--
-- The design is deliberately /gradual/: the backing tensor keeps the static
-- shape @maxLen ': shape@, and only the number of filled slots is a runtime
-- value, kept consistent by this module's smart constructors ('append'
-- refuses to overfill).  One dimension gives up static checking; everything
-- around it keeps it.  There are no proofs here and none are needed — the
-- invariant is three fields wide.
--
-- Downstream code treats the buffer as the full @maxLen@ tensor plus a mask:
-- 'attentionMask' is @0@ over the filled prefix and @-∞@ over the empty
-- tail, so masked attention over the buffer equals attention over the true
-- prefix — the equation the test suite checks.
module Torch.Typed.Buffer
  ( Buffer,
    emptyBuffer,
    append,
    used,
    capacity,
    bufferTensor,
    validMask,
    attentionMask,
  )
where

import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional.Internal as I
import qualified Torch.Tensor as D
import Torch.Typed.Auxiliary (natValI)
import Torch.Typed.Factories (zeros)
import Torch.Typed.Tensor

-- | @Buffer device dtype maxLen shape@ holds up to @maxLen@ entries of shape
-- @shape@.  Unfilled slots read as zeros.
data Buffer (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (maxLen :: Nat) (shape :: [Nat]) = UnsafeMkBuffer
  { -- | The backing tensor: filled prefix, zeros after it.
    bufferTensor :: Tensor device dtype (maxLen ': shape),
    -- | How many slots are filled, in @[0 .. maxLen]@.
    used :: Int
  }

instance Show (Buffer device dtype maxLen shape) where
  show b = "Buffer(" <> show (used b) <> " used) " <> show (bufferTensor b)

-- | The static capacity, reified.
capacity :: forall device dtype maxLen shape. KnownNat maxLen => Buffer device dtype maxLen shape -> Int
capacity _ = natValI @maxLen

-- | An empty buffer of zeros.
emptyBuffer ::
  forall maxLen shape device dtype.
  TensorOptions (maxLen ': shape) dtype device =>
  Buffer device dtype maxLen shape
emptyBuffer = UnsafeMkBuffer zeros 0

-- | Append one entry.  'Nothing' when the buffer is full — the one place the
-- runtime length meets the static bound.
append ::
  forall device dtype maxLen shape.
  KnownNat maxLen =>
  Tensor device dtype shape ->
  Buffer device dtype maxLen shape ->
  Maybe (Buffer device dtype maxLen shape)
append x (UnsafeMkBuffer buf n)
  | n >= natValI @maxLen = Nothing
  | otherwise =
      Just $
        UnsafeMkBuffer
          ( UnsafeMkTensor $
              I.slice_scatter
                (toDynamic buf)
                (I.unsqueeze (toDynamic x) 0)
                0
                n
                (n + 1)
                1
          )
          (n + 1)

-- | @True@ over the filled prefix, @False@ over the tail.
validMask ::
  forall device dtype maxLen shape.
  (KnownNat maxLen, KnownDevice device) =>
  Buffer device dtype maxLen shape ->
  Tensor device 'D.Bool '[maxLen]
validMask (UnsafeMkBuffer _ n) =
  UnsafeMkTensor . D.toDevice (deviceVal @device) . D.asTensor $
    [i < n | i <- [0 .. natValI @maxLen - 1]]

-- | @0@ over the filled prefix, @-∞@ over the tail: add it to attention
-- scores so the empty slots get zero weight after softmax.
attentionMask ::
  forall device dtype maxLen shape.
  (KnownNat maxLen, KnownDevice device) =>
  Buffer device dtype maxLen shape ->
  Tensor device dtype '[maxLen]
attentionMask (UnsafeMkBuffer buf n) =
  UnsafeMkTensor
    . D.toDevice (deviceVal @device)
    . D.toType (D.dtype (toDynamic buf))
    . D.asTensor
    $ [if i < n then 0 else -1 / 0 :: Float | i <- [0 .. natValI @maxLen - 1]]
