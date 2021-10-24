{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Comparison where

import Control.Monad.Catch (MonadThrow)
import Data.Singletons (SingI (..), SingKind (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, GetDimImplF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SSelectDim, SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3)
import Torch.Internal.GC (unsafeThrowableIO)
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
    forall gradient layout device dataType shape gradient' layout' device' dataType' shape' shape'' m.
    (MonadThrow m, Catch (dataType <+> dataType'), shape'' ~ BroadcastShapesF shape shape', Catch shape'') =>
    Tensor gradient layout device dataType shape ->
    Tensor gradient' layout' device' dataType' shape' ->
    m
      ( Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> layout')
          (device <+> device')
          ('DataType 'Bool)
          shape''
      )
a `gt` b = unsafeThrowableIO $ cast2 ATen.gt_tt a b
a `lt` b = unsafeThrowableIO $ cast2 ATen.lt_tt a b
a `ge` b = unsafeThrowableIO $ cast2 ATen.ge_tt a b
a `le` b = unsafeThrowableIO $ cast2 ATen.le_tt a b
a `eq` b = unsafeThrowableIO $ cast2 ATen.eq_tt a b
a `ne` b = unsafeThrowableIO $ cast2 ATen.ne_tt a b
(>.) = gt
(<.) = lt
(>=.) = ge
(<=.) = le
(==.) = eq
(/=.) = ne

data Order = Ascending | Descending
  deriving stock (Show, Eq, Ord, Generic)

data Sorted gradient layout device dataType shape where
  Sorted ::
    forall gradient layout device dataType shape.
    { sorted :: Tensor gradient layout device dataType shape,
      indices :: Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int64) shape
    } ->
    Sorted gradient layout device dataType shape
  deriving stock (Show, Generic)

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
  forall selectDim gradient layout device dataType shape.
  SSelectDim selectDim ->
  Order ->
  Tensor gradient layout device dataType shape ->
  Sorted gradient layout device dataType (SortF selectDim shape)
sSort by order tensor =
  let by' = forgetIsChecked $ fromSing by
   in uncurry Sorted $ case by' of
        ByName name -> unsafePerformIO $ cast3 ATen.sort_tnb tensor name (order == Descending)
        ByIndex index -> unsafePerformIO $ cast3 ATen.sort_tlb tensor (fromInteger index :: Int) (order == Descending)

sort ::
  forall selectDim gradient layout device dataType shape.
  SingI selectDim =>
  Order ->
  Tensor gradient layout device dataType shape ->
  Sorted gradient layout device dataType (SortF selectDim shape)
sort = sSort (sing @selectDim)
