{-# LANGUAGE LambdaCase #-}
module Torch.Core.Internal
  ( w2cl
  , i2cl
  , onDims
  , showLim
  ) where

import Foreign (Word, Ptr)
import Foreign.C.Types (CLong)
import TensorTypes (TensorDim(..))
import Numeric (showGFloat)

w2cl :: Word -> CLong
w2cl = fromIntegral

i2cl :: Integer -> CLong
i2cl = fromIntegral

-- | simple helper to clean up common pattern matching on TensorDim
onDims
  :: (a0 -> a)
  -> b
  -> ( a -> b )
  -> ( a -> a -> b )
  -> ( a -> a -> a -> b )
  -> ( a -> a -> a -> a -> b )
  -> TensorDim a0
  -> b
onDims ap f0 f1 f2 f3 f4 = \case
  D0 -> f0
  D1 d1 -> f1 (ap d1)
  D2 (d1, d2) -> f2 (ap d1) (ap d2)
  D3 (d1, d2, d3) -> f3 (ap d1) (ap d2) (ap d3)
  D4 (d1, d2, d3, d4) -> f4 (ap d1) (ap d2) (ap d3) (ap d4)


-- | Show a real value with limited precision
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

