{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Dimensions
  ( module X
  , KnownNat
  , KnownNat2
  , KnownNat3
  , KnownNatDim
  , KnownNatDim2
  , KnownNatDim3
  , SingDim
  , SingDim2
  , SingDim3
  , SingDimensions
  , DimVal(..)
  , someDimsM
  , unsafeSomeDims
  , showdim
  , showdim'
  , rankCheck
  , onDims
  , onDims'
  , dimVals
  , dimVals'
  , rank'
  , product'
  , module Dim
  ) where

import Data.Singletons as X
import Data.Singletons.Prelude.List as X
  hiding (Take, Tail, Reverse, Last, Init, Head, Length, Drop, Concat)

import Data.Proxy (Proxy(..))
import Control.Monad (unless)
import Control.Exception.Safe (throwString, MonadThrow)
import Data.Foldable (toList)
import Data.Maybe (fromMaybe)
import Data.List (intercalate)
import Data.Sequence (Seq, (|>))
import GHC.TypeLits (KnownNat)
import Numeric.Dimensions (Dim(..), SomeDims(..), Nat)
import Numeric.Dimensions as Dim
import GHC.Int (Int32)

type KnownNat2 n0 n1    = (KnownNat n0, KnownNat n1)
type KnownNat3 n0 n1 n2 = (KnownNat n0, KnownNat n1, KnownNat n2)

type KnownNatDim n         = (KnownDim n, KnownNat n)
type KnownNatDim2 n0 n1    = (KnownNatDim n0, KnownNatDim n1)
type KnownNatDim3 n0 n1 n2 = (KnownNatDim n0, KnownNatDim n1, KnownNatDim n2)
type SingDim n         = (SingI n, KnownNatDim n)
type SingDim2 n0 n1    = (SingDim n0, SingDim n1)
type SingDim3 n0 n1 n2 = (SingDim n0, SingDim n1, SingDim n2)

type SingDimensions d = (SingI d, Dimensions d)

newtype DimVal = DimVal Int32
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

someDimsM :: (MonadThrow m, Integral i) => [i] -> m SomeDims
someDimsM d = case Dim.someDimsVal (fmap fromIntegral d) of
  Nothing -> throwString "User Defined Error: included dimension of size 0, review tensor dimensionality."
  Just sd -> pure sd

unsafeSomeDims :: [Int] -> SomeDims
unsafeSomeDims d = fromMaybe impureError (Dim.someDimsVal d)
 where
  impureError = error "User Defined Error: included dimension of size 0, review tensor dimensionality."

{-
transferDims :: Proxy (ds::[Nat]) -> Dim ds
transferDims p = undefined
 where

go :: forall f m . Proxy (m::[Nat]) -> Dim (f :: [Nat])
go _ =
  if null (fromSing (sing :: Sing m))
  then (D  :: Dim f)
  else (Dn :: (x:xs) ~ m => Dim (x::Nat)) :* (go (Proxy :: (x:xs) ~ m => Proxy xs))
-- -}

showdim :: Dim (ds::[k]) -> String
showdim = go mempty
  where
    printlist :: Seq String -> String
    printlist acc = "TensorDim [" ++ intercalate "," (toList acc) ++ "]"

    showScalar :: Dim (ds' :: k) -> String
    showScalar (   d@Dn) =         show (Dim.dimVal d)
    showScalar (Dx d@Dn) = ">=" ++ show (Dim.dimVal d) -- primarily for this rendering change
    showScalar _ = error "only to be called on scalars"

    go :: Seq String -> Dim (ds::[k]) -> String
    go acc       D   = printlist acc
    go acc (d :* D)  = printlist (acc |> showScalar d)
    go acc (d :* ds) = go (acc |> showScalar d) ds

showdim' :: SomeDims -> String
showdim' (SomeDims ds) = showdim ds

-- | Runtime type-level check of # dimensions, ported from core/src/generic/Torch/Core/Static/Double.hs
-- TODO: possibly remove this function?
rankCheck :: MonadThrow m => SomeDims -> Int -> m ()
rankCheck dims n
  = unless
    (length (dimVals' dims) == n)
    (throwString "Incorrect Dimensions")

-- Helper function to debug dimensions package. We return @Integral i@ in case we need to cast directly to C-level types.
dimVals :: Integral i => Dim (ns::[k]) -> [i]
dimVals = go mempty
  where
    go :: forall i ns . Integral i => Seq i -> Dim (ns::[k]) -> [i]
    go acc D         = toList   acc
    go acc (d :* D)  = toList $ acc |> fromIntegral (Dim.dimVal d)
    go acc (d :* ds) = go (acc |> fromIntegral (Dim.dimVal d)) ds

dimVals' :: SomeDims -> [Int]
dimVals' (SomeDims ds) = dimVals ds

rank' :: SomeDims -> Int
rank' = length . dimVals'

product' :: SomeDims -> Int
product' (SomeDims d) = product (dimVals d)


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
onDims ap f0 f1 f2 f3 f4 dim = case dimVals dim of
  [] -> f0
  [d1]  -> f1 (ap d1)
  [d1, d2] -> f2 (ap d1) (ap d2)
  [d1, d2, d3] -> f3 (ap d1) (ap d2) (ap d3)
  [d1, d2, d3, d4] -> f4 (ap d1) (ap d2) (ap d3) (ap d4)
  _ -> error "impossible pattern match"


-- | simple helper to clean up common pattern matching on TensorDim
onDims'
  :: (Int -> a)
  -> b
  -> ( a -> b )
  -> ( a -> a -> b )
  -> ( a -> a -> a -> b )
  -> ( a -> a -> a -> a -> b )
  -> SomeDims
  -> b
onDims' ap f0 f1 f2 f3 f4 (SomeDims ds) =
  onDims ap f0 f1 f2 f3 f4 ds
