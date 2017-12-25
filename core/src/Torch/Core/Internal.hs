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

  , impossible
  ) where

import Foreign (Word, Ptr)
import Foreign.C.Types (CLLong, CLong, CDouble, CShort, CLong, CChar, CInt, CFloat)
import THTypes
import Numeric (showGFloat)

import Torch.Core.Tensor.Dim

w2cll :: Word -> CLLong
w2cll = fromIntegral

w2cl :: Word -> CLong
w2cl = fromIntegral

i2cl :: Integer -> CLong
i2cl = fromIntegral

i2cll :: Integer -> CLLong
i2cll = fromIntegral

fi = fromIntegral

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

-- ========================================================================= --

impossible :: String -> a
impossible = error
