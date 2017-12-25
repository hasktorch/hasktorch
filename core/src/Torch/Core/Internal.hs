{-# LANGUAGE GADTs #-}
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

  , impossible
  ) where

import Foreign (Word, Ptr)
import Foreign.C.Types (CLLong, CLong, CDouble)
import Numeric (showGFloat)
import Numeric.Dimensions (Dim(..))
import Data.List (intercalate)
import GHC.TypeLits(natVal)
import Data.Sequence (Seq, (|>))
import Data.Foldable (toList)
import Control.Monad (fail)
import Data.Exceptions.Safe

import qualified Numeric.Dimensions as Dim

w2cll :: Word -> CLLong
w2cll = fromIntegral

w2cl :: Word -> CLong
w2cl = fromIntegral

i2cl :: Integer -> CLong
i2cl = fromIntegral

i2cll :: Integer -> CLLong
i2cll = fromIntegral

fi = fromIntegral

data DimView
  = D0
  | D1  Int
  | D2  Int Int
  | D3  Int Int Int
  | D4  Int Int Int Int
  | D5  Int Int Int Int Int
  | D6  Int Int Int Int Int Int
  | D7  Int Int Int Int Int Int Int
  | D8  Int Int Int Int Int Int Int Int
  | D9  Int Int Int Int Int Int Int Int Int
  | D10 Int Int Int Int Int Int Int Int Int Int

showdim :: Dim ns -> String
showdim = go mempty
  where
    printlist :: Seq String -> String
    printlist acc = "TensorDim: [" ++ intercalate "," (toList acc) ++ "]"

    go :: Seq String -> Dim ns -> String
    go acc D         = printlist acc
    go acc d@Dn      = printlist (acc |> show (dimVal d))
    go acc (Dx d@Dn) = printlist (acc |> (">=" ++ show (dimVal d)))  -- primarily for this rendering change
    go acc (d :* D)  = printlist (acc |> show (dimVal d))
    go acc (d :* ds) = go (acc |> show (dimVal d)) ds

-- Helper function to debug dimensions package
dimvals :: Dim ns -> [Int]
dimvals = go mempty
  where
    go :: Seq Int -> Dim ns -> [Int]
    go acc D         = toList   acc
    go acc d@Dn      = toList $ acc |> dimVal d
    go acc (Dx d@Dn) = toList $ acc |> dimVal d
    go acc (d :* D)  = toList $ acc |> dimVal d
    go acc (d :* ds) = go (acc |> dimVal d) ds

view :: Dim a -> DimView
view d = case dimVals d of
  [] -> D0
  [d1] -> D1 d1
  [d1,d2] -> D2 d1 d2
  [d1,d2,d3] -> D3 d1 d2 d3
  [d1,d2,d3,d4] -> D4 d1 d2 d3 d4
  [d1,d2,d3,d4,d5] -> D5 d1 d2 d3 d4 d5
  [d1,d2,d3,d4,d5,d6] -> D6 d1 d2 d3 d4 d5 d6
  [d1,d2,d3,d4,d5,d6,d7] -> D7 d1 d2 d3 d4 d5 d6 d7
  [d1,d2,d3,d4,d5,d6,d7,d8] -> D8 d1 d2 d3 d4 d5 d6 d7 d8
  [d1,d2,d3,d4,d5,d6,d7,d8,d9] -> D9 d1 d2 d3 d4 d5 d6 d7 d8 d9
  [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10] -> D10 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10
  _  -> error "tensor rank is not accounted for in the view pattern"

-- | simple helper to clean up common pattern matching on TensorDim
onDims
  :: (Int -> a)
  -> b
  -> ( a -> b )
  -> ( a -> a -> b )
  -> ( a -> a -> a -> b )
  -> ( a -> a -> a -> a -> b )
  -> Dim dims
  -> b
onDims ap f0 f1 f2 f3 f4 dim = case view dim of
  D0 -> f0
  D1 d1 -> f1 (ap d1)
  D2 d1 d2 -> f2 (ap d1) (ap d2)
  D3 d1 d2 d3 -> f3 (ap d1) (ap d2) (ap d3)
  D4 d1 d2 d3 d4 -> f4 (ap d1) (ap d2) (ap d3) (ap d4)



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
