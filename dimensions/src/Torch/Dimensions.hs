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
{-# LANGUAGE CPP #-}
module Torch.Dimensions
  ( module X
  , KnownNat -- ,   SingDim
  , KnownNat2, KnownDim2, Dimensions2
  , KnownNat3, KnownDim3, Dimensions3
  , KnownNat4, KnownDim4, Dimensions4
  , KnownNat5, KnownDim5, Dimensions5
  , SingDimensions
  , DimVal(..)
  , module Dim
  ) where

import Data.Singletons as X

import Data.Singletons.Prelude.Bool as X
import Data.Singletons.Prelude.Num as X
#if MIN_VERSION_singletons(2,1,0)
  hiding (type (:+), type (:<))
#endif

import Data.Singletons.Prelude.Ord as X
#if MIN_VERSION_singletons(2,1,0)
  hiding (type (:+), type (:<))
#endif

import Data.Singletons.Prelude.List as X
  hiding (Take, Tail, Reverse, Last, Init, Head, Length, Drop, Concat, type (++))
import Data.Singletons.TypeLits as X
  hiding (KnownNat)

import Data.Proxy as X (Proxy(..))
import Control.Monad (unless)
import Control.Exception.Safe (throwString, MonadThrow)
import Data.Foldable (toList)
import Data.Maybe (fromMaybe)
import Data.List (intercalate)
import Data.Sequence (Seq, (|>))
import GHC.TypeLits (KnownNat)
import Numeric.Dimensions (Dim(..), SomeDims(..), Nat)
import Numeric.Dimensions as Dim hiding
  (type (+), type (*), All, type (-), type Map)
import GHC.Int (Int32)

type KnownNat2 n0 n1       = (KnownNat  n0,       KnownNat n1)
type KnownNat3 n0 n1 n2    = (KnownNat2 n0 n1,    KnownNat n2)
type KnownNat4 n0 n1 n2 n3 = (KnownNat3 n0 n1 n2, KnownNat n3)
type KnownNat5 n0 n1 n2 n3 n4 = (KnownNat4 n0 n1 n2 n3, KnownNat n4)

type KnownDim2 n0 n1       = (KnownDim  n0,       KnownDim n1)
type KnownDim3 n0 n1 n2    = (KnownDim2 n0 n1,    KnownDim n2)
type KnownDim4 n0 n1 n2 n3 = (KnownDim3 n0 n1 n2, KnownDim n3)
type KnownDim5 n0 n1 n2 n3 n4 = (KnownDim4 n0 n1 n2 n3, KnownDim n4)

type SingDim n            = (SingI n, KnownDim n)
type SingDim2 n0 n1       = (SingDim n0,        SingDim n1)
type SingDim3 n0 n1 n2    = (SingDim2 n0 n1,    SingDim n2)
type SingDim4 n0 n1 n2 n3 = (SingDim3 n0 n1 n2, SingDim n3)
type SingDim5 n0 n1 n2 n3 n4 = (SingDim4 n0 n1 n2 n3, SingDim n4)

type SingDimensions d = (SingI d, Dimensions d)

type (Dimensions2 d d') = (Dimensions d, Dimensions d')
type (Dimensions3 d d' d'' ) = (Dimensions2 d d', Dimensions d'')
type (Dimensions4 d d' d'' d''') = (Dimensions2 d d', Dimensions2 d'' d''')
type (Dimensions5 d d' d'' d''' d'''') = (Dimensions4 d d' d'' d''', Dimensions d'''')


newtype DimVal = DimVal Int32
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

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

-- Helper function to debug dimensions package. We return @Integral i@ in case we need to cast directly to C-level types.

product' :: SomeDims -> Int
product' (SomeDims d) = fromIntegral $ totalDim d


