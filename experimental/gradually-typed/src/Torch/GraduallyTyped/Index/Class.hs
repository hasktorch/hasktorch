{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Index.Class where

import Data.Kind (Constraint)
import Data.Type.Equality (type (==))
import GHC.TypeLits (CmpNat, Nat, Symbol, TypeError)
import Torch.GraduallyTyped.Index.Type (Index (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name, Size (..))
import Type.Errors.Pretty (type (<>))

type family IndexOutOfBound (idx :: Nat) (dim :: Dim (Name Symbol) (Size Nat)) where
  IndexOutOfBound idx dim =
    TypeError ("Out of bound index " <> idx <> " for dimension " <> dim)

type family InRangeImplF (idx :: Index Nat) (dim :: Dim (Name Symbol) (Size Nat)) :: Bool where
  InRangeImplF 'UncheckedIndex _ = 'True
  InRangeImplF _ ('Dim _ 'UncheckedSize) = 'True
  InRangeImplF ('Index idx) ('Dim _ ('Size index)) = CmpNat idx index == 'LT

type family InRangeCheckF (idx :: Index Nat) (dim :: Dim (Name Symbol) (Size Nat)) (ok :: Bool) :: Constraint where
  InRangeCheckF _ _ 'True = ()
  InRangeCheckF ('Index idx) dim _ = IndexOutOfBound idx dim

type InRangeF idx dim = InRangeCheckF idx dim (InRangeImplF idx dim)
