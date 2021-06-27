{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Comparison where

import Data.Singletons (SingI (..), SingKind (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Prelude (Seq, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, GetDimImplF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SSelectDim, SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tuple as ATen ()
import Type.Errors.Pretty (TypeError, type (%), type (<>))

gt,
  lt,
  ge,
  le,
  eq,
  ne,
  (>.),
  (<.),
  (>=.),
  (<=.),
  (==.),
  (/=.) ::
    forall requiresGradient layout device dataType shape requiresGradient' layout' device' dataType' shape'.
    Tensor requiresGradient layout device dataType shape ->
    Tensor requiresGradient' layout' device' dataType' shape' ->
    Tensor
      'WithoutGradient
      (layout <+> layout')
      (device <+> device')
      (Seq (dataType <+> dataType') ('DataType 'Bool))
      (BroadcastShapesF shape shape')
a `gt` b = unsafePerformIO $ cast2 ATen.gt_tt a b
a `lt` b = unsafePerformIO $ cast2 ATen.lt_tt a b
a `ge` b = unsafePerformIO $ cast2 ATen.ge_tt a b
a `le` b = unsafePerformIO $ cast2 ATen.le_tt a b
a `eq` b = unsafePerformIO $ cast2 ATen.eq_tt a b
a `ne` b = unsafePerformIO $ cast2 ATen.ne_tt a b
(>.) = gt
(<.) = lt
(>=.) = ge
(<=.) = le
(==.) = eq
(/=.) = ne

data Order = Ascending | Descending deriving stock (Show, Eq, Ord, Generic)

data Sorted requiresGradient layout device dataType shape where
  Sorted ::
    forall requiresGradient layout device dataType shape.
    { sorted :: Tensor requiresGradient layout device dataType shape,
      indices :: Tensor 'WithoutGradient layout device ('DataType 'Int64) shape
    } ->
    Sorted requiresGradient layout device dataType shape

type SortErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply sort on the dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family SortCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe (Dim (Name Symbol) (Size Nat))) :: [Dim (Name Symbol) (Size Nat)] where
  SortCheckF by dims 'Nothing = TypeError (SortErrorMessage by dims)
  SortCheckF _ dims ('Just _) = dims

type family SortF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  SortF 'UncheckedSelectDim _ = 'UncheckedShape
  SortF _ 'UncheckedShape = 'UncheckedShape
  SortF ('SelectDim by) ('Shape dims) = 'Shape (SortCheckF by dims (GetDimImplF by dims))

sSort ::
  forall selectDim requiresGradient layout device dataType shape.
  SSelectDim selectDim ->
  Order ->
  Tensor requiresGradient layout device dataType shape ->
  Sorted requiresGradient layout device dataType (SortF selectDim shape)
sSort by order tensor =
  let by' = forgetIsChecked $ fromSing by
   in uncurry Sorted $ case by' of
        ByName name -> unsafePerformIO $ cast3 ATen.sort_tnb tensor name (order == Descending)
        ByIndex index -> unsafePerformIO $ cast3 ATen.sort_tlb tensor (fromInteger index :: Int) (order == Descending)

sort ::
  forall selectDim requiresGradient layout device dataType shape.
  SingI selectDim =>
  Order ->
  Tensor requiresGradient layout device dataType shape ->
  Sorted requiresGradient layout device dataType (SortF selectDim shape)
sort = sSort (sing @selectDim)