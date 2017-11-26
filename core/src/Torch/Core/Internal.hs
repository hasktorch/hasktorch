{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Core.Internal
  ( w2cl
  , w2cll
  , i2cl
  , i2cll
  , fromIntegral
  , onDims
  , showLim
  , genOp1
  , genOp2

  , Positive
  , mkPositive
  , fromPositive
  ) where

import Foreign (Word, Ptr)
import Foreign.C.Types (CLLong, CLong, CDouble)
import Torch.Core.Tensor.Types (TensorDim(..))
import Numeric (showGFloat)

w2cll :: Word -> CLLong
w2cll = fromIntegral

w2cl :: Word -> CLong
w2cl = fromIntegral

i2cl :: Integer -> CLong
i2cl = fromIntegral

i2cll :: Integer -> CLLong
i2cll = fromIntegral

fi = fromIntegral

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

-- | generic function for 1-arity, monomorphic CDouble functions
genOp1 :: (Real a, Fractional b) => (CDouble -> CDouble) -> a -> b
genOp1 thop a = realToFrac $ thop (realToFrac a)

-- | generic function for 2-arity, monomorphic CDouble functions
genOp2 :: (Real a, Fractional b) => (CDouble -> CDouble -> CDouble) -> a -> a -> b
genOp2 thop a b = realToFrac $ thop (realToFrac a) (realToFrac b)


-- ========================================================================= --

newtype Positive n = Positive { unPositive :: n }
  deriving (Eq, Ord, Show, Num, Fractional)

mkPositive :: (Ord n, Num n) => n -> Maybe (Positive n)
mkPositive n
  | n < 0 = Nothing
  | otherwise = Just (Positive n)

fromPositive :: Positive n -> n
fromPositive = unPositive

