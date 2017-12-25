{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeInType #-}

module Torch.Core.Tensor.Dim
  ( DimView(..)
  -- , showdim
  , onDims
  , dimVals
  , view
  ) where

-- import Control.Exception.Safe
import Data.Foldable (toList)
import Data.List (intercalate)
import Data.Sequence (Seq, (|>))
-- import GHC.TypeLits(natVal)
import Numeric.Dimensions (Dim(..))

import qualified Numeric.Dimensions as Dim

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

showdim :: Dim (ds::[k]) -> String
showdim = go mempty
  where
    printlist :: Seq String -> String
    printlist acc = "TensorDim [" ++ intercalate "," (toList acc) ++ "]"

    showScalar :: Dim (ds' :: k) -> String
    showScalar (   d@Dn) =         show (Dim.dimVal d)
    showScalar (Dx d@Dn) = ">=" ++ show (Dim.dimVal d) -- primarily for this rendering change

    go :: Seq String -> Dim (ds::[k]) -> String
    go acc       D   = printlist acc
    go acc (d :* D)  = printlist (acc |> showScalar d)
    go acc (d :* ds) = go (acc |> showScalar d) ds

-- Helper function to debug dimensions package
dimVals :: Dim (ns::[k]) -> [Int]
dimVals = go mempty
  where
    go :: Seq Int -> Dim (ns::[k]) -> [Int]
    go acc D         = toList   acc
    go acc (d :* D)  = toList $ acc |> Dim.dimVal d
    go acc (d :* ds) = go (acc |> Dim.dimVal d) ds

view :: Dim (ns::[k]) -> DimView
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
  _  -> error "tensor rank is not accounted for in view pattern"

-- | should be up to rank 5, these are the "most common" cases
view4 :: Dim (ns::[k]) -> DimView
view4 d = case view d of
  d@D0   -> d
  d@D1{} -> d
  d@D2{} -> d
  d@D3{} -> d
  d@D4{} -> d
  _  -> error "tensor rank is not accounted for in view pattern"


-- | simple helper to clean up common pattern matching on TensorDim
onDims
  :: (Int -> a)
  -> b
  -> ( a -> b )
  -> ( a -> a -> b )
  -> ( a -> a -> a -> b )
  -> ( a -> a -> a -> a -> b )
  -> Dim (dims::[k])
  -> b
onDims ap f0 f1 f2 f3 f4 dim = case view4 dim of
  D0 -> f0
  D1 d1 -> f1 (ap d1)
  D2 d1 d2 -> f2 (ap d1) (ap d2)
  D3 d1 d2 d3 -> f3 (ap d1) (ap d2) (ap d3)
  D4 d1 d2 d3 d4 -> f4 (ap d1) (ap d2) (ap d3) (ap d4)
  _ -> error "impossible pattern match"


