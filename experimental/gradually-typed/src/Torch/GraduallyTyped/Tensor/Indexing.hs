{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}

module Torch.GraduallyTyped.Tensor.Indexing where

import Control.Arrow ((>>>))
import Control.Monad (forM_)
import Control.Monad.Catch (MonadThrow)
import Data.Coerce (coerce)
import Data.Kind (Type)
import Data.Singletons (Demote, SingI, SingKind, SomeSing (..), fromSing, sing, toSing, withSomeSing)
import Data.Singletons.Prelude.List (Reverse, SList, Sing)
import Data.Singletons.TH (genSingletons)
import Data.Type.Equality (type (==))
import Foreign (fromBool)
import GHC.TypeLits (Div, ErrorMessage (..), Nat, Symbol, type (+), type (-), type (<=?))
import Torch.GraduallyTyped.Index (DemotedIndex (..), Index (..))
import Torch.GraduallyTyped.Prelude (If, IsChecked (..), forgetIsChecked, type (<?))
import Torch.GraduallyTyped.Shape.Class (PrependDimF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import Type.Errors.Pretty (TypeError, type (%), type (<>))

data IndexType a
  = None
  | Ellipsis
  | SliceAll
  | SliceAt a
  | SliceBool Bool
  | SliceFrom a
  | SliceUpTo a
  | SliceFromUpTo a a
  | SliceFromWithStride a a
  | SliceUpToWithStride a a
  | SliceFromUpToWithStride a a a
  deriving (Functor)

genSingletons [''IndexType]

type ReverseShape :: Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family ReverseShape shape where
  ReverseShape 'UncheckedShape = 'UncheckedShape
  ReverseShape ('Shape dims) = 'Shape (Reverse dims)

type ErrorOnEllipsis :: [IndexType (Index Nat)] -> [IndexType (Index Nat)]
type family ErrorOnEllipsis indices where
  ErrorOnEllipsis '[] = '[]
  ErrorOnEllipsis ('Ellipsis ': ixs) = TypeError ('Text "Indices can only contain a single ellipsis.")
  ErrorOnEllipsis (ix ': ixs) = ix ': ErrorOnEllipsis ixs

type StrideZeroErrorMessage = 'Text "Slice stride cannot be zero"

-- | Calculate the size of the dimension from stride.
--
-- >>> :kind! Strided 8 1
-- Strided 8 1 :: Nat
-- = 8
-- >>> :kind! Strided 5 2
-- Strided 5 2 :: Nat
-- = 3
-- >>> :kind! Strided 6 3
-- Strided 6 3 :: Nat
-- = 2
type Strided :: Nat -> Nat -> Nat
type family Strided length stride where
  Strided _ 0 = TypeError StrideZeroErrorMessage
  Strided 0 _ = 0
  Strided length stride = ((length - 1) `Div` stride + 1)

type family CheckUpTo (upTo :: Nat) ok where
  CheckUpTo upTo ok =
    If
      (upTo == 0)
      (TypeError ('Text "Slice 'upTo' type parameter must not be equal to zero"))
      ok

type family CheckFromSize (from :: Nat) (size :: Nat) ok where
  CheckFromSize from size ok =
    If
      (from <? size)
      ok
      ( TypeError
          ( "Slice 'from' type parameter must be smaller than the size of the indexed dimension:"
              % "    " <> "from < size"
              % "but"
              % "    " <> from <> " >= " <> size
          )
      )

type family CheckUpToSize (upTo :: Nat) (size :: Size Nat) ok where
  CheckUpToSize upTo 'UncheckedSize ok = CheckUpTo upTo ok
  CheckUpToSize upTo ('Size size) ok =
    CheckUpTo
      upTo
      ( If
          (upTo <=? size)
          ok
          ( TypeError
              ( "Slice 'upTo' type parameter must be less than or equal to the size of the indexed dimension:"
                  % "    " <> "upTo <= size"
                  % "but"
                  % "    " <> upTo <> " > " <> size
              )
          )
      )

type family CheckFromUpTo (from :: Nat) (upTo :: Nat) ok where
  CheckFromUpTo from upTo ok =
    If
      (from <? upTo)
      ok
      ( TypeError
          ( "Slice 'from' type parameter must be less than the 'upTo' type parameter:"
              % "    " <> "from < upTo"
              % "but"
              % "    " <> from <> " >= " <> upTo
          )
      )

type family CheckFromUpToSize (from :: Nat) (upTo :: Nat) (size :: Size Nat) ok where
  CheckFromUpToSize from upTo size ok = CheckFromUpTo from upTo (CheckUpToSize upTo size ok)

type family CheckSliceAt (at :: Nat) (size :: Nat) ok where
  CheckSliceAt at size ok =
    If
      (at <? size)
      ok
      ( TypeError
          ( "Index of 'SliceAt " <> at <> "' must be less than the dimension size:"
              % "    " <> "at < size"
              % "but"
              % "    " <> at <> " >= " <> size
          )
      )

type IndexDimsImpl :: [IndexType (Index Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family IndexDimsImpl indices dims where
  IndexDimsImpl '[] dims = 'Shape dims
  IndexDimsImpl ('None ': ixs) dims = 'Dim ('Name "*") ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('Ellipsis ': _) '[] = 'Shape '[]
  IndexDimsImpl ('Ellipsis ': ixs) dims = ReverseShape (IndexDimsImpl (Reverse (ErrorOnEllipsis ixs)) (Reverse dims))
  IndexDimsImpl ('SliceAll ': ixs) (dim ': dims) = dim `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceAt ('Index at) ': ixs) ('Dim name ('Size size) ': dims) = CheckSliceAt at size (IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceAt _ ': ixs) ('Dim name _ ': dims) = IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'False ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 0) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'True ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFrom ('Index from) ': ixs) ('Dim name ('Size size) ': dims) =
    CheckFromSize from size ('Dim name ('Size (size - from)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFrom _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpTo ('Index upTo) ': ixs) ('Dim name size ': dims) =
    CheckUpToSize upTo size ('Dim name ('Size upTo) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceUpTo _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpTo ('Index from) ('Index upTo) ': ixs) ('Dim name size ': dims) =
    CheckFromUpToSize from upTo size ('Dim name ('Size (upTo - from)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpTo _ _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStride ('Index from) ('Index stride) ': ixs) ('Dim name ('Size size) ': dims) =
    CheckFromSize from size ('Dim name ('Size (Strided (size - from) stride)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromWithStride _ _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpToWithStride ('Index upTo) ('Index stride) ': ixs) ('Dim name size ': dims) =
    CheckUpToSize upTo size ('Dim name ('Size (Strided upTo stride)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceUpToWithStride _ _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride ('Index from) ('Index upTo) ('Index stride) ': ixs) ('Dim name size ': dims) =
    CheckFromUpToSize from upTo size ('Dim name ('Size (Strided (upTo - from) stride)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpToWithStride _ _ _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims

type family IndexDims indices shape where
  IndexDims 'UncheckedIndices _ = 'UncheckedShape
  IndexDims _ 'UncheckedShape = 'UncheckedShape
  IndexDims ('Indices indices) ('Shape dims) = IndexDimsImpl indices dims

data Indices (dimIndices :: Type) where
  UncheckedIndices :: forall dimIndices. Indices dimIndices
  Indices :: forall dimIndices. dimIndices -> Indices dimIndices

data SIndices (indices :: Indices [IndexType (Index Nat)]) where
  SUncheckedIndices :: [IndexType Integer] -> SIndices 'UncheckedIndices
  SIndices :: forall dimIndices. SList dimIndices -> SIndices ('Indices dimIndices)

type instance Sing = SIndices

instance SingI dimIndices => SingI ('Indices (dimIndices :: [IndexType (Index Nat)])) where
  sing = SIndices $ sing @dimIndices

instance SingKind (Indices [IndexType (Index Nat)]) where
  type Demote (Indices [IndexType (Index Nat)]) = IsChecked [IndexType (IsChecked Integer)]
  fromSing (SUncheckedIndices indexTypes) = Unchecked $ fmap Unchecked <$> indexTypes
  fromSing (SIndices indexTypes) = Checked . coerce . fromSing $ indexTypes
  toSing (Unchecked indexTypes) = SomeSing . SUncheckedIndices $ fmap forgetIsChecked <$> indexTypes
  toSing (Checked indexTypes) = withSomeSing ((fmap . fmap . fmap) DemotedIndex indexTypes) $ SomeSing . SIndices

(!) ::
  forall indices requiresGradient layout device dataType shape m.
  MonadThrow m =>
  Tensor requiresGradient layout device dataType shape ->
  SIndices indices ->
  m (Tensor requiresGradient layout device dataType (IndexDims indices shape))
(UnsafeTensor t) ! sIndices = unsafeThrowableIO $ do
  indexList <- ATen.newTensorIndexList
  tensorIndices <- traverse toTensorIndex indices
  forM_ tensorIndices $ ATen.tensorIndexList_push_back indexList
  UnsafeTensor <$> ATen.index t indexList
  where
    indices = fmap forgetIsChecked <$> forgetIsChecked (fromSing sIndices)
    toTensorIndex =
      fmap fromIntegral >>> \case
        None -> ATen.newTensorIndexWithNone
        Ellipsis -> ATen.newTensorIndexWithEllipsis
        SliceAt at -> ATen.newTensorIndexWithInt at
        SliceBool b -> ATen.newTensorIndexWithBool (fromBool b)
        SliceAll -> ATen.newTensorIndexWithSlice 0 maxBound 1
        SliceFrom from -> ATen.newTensorIndexWithSlice from maxBound 1
        SliceUpTo upTo -> ATen.newTensorIndexWithSlice 0 upTo 1
        SliceFromUpTo from upTo -> ATen.newTensorIndexWithSlice from upTo 1
        SliceFromWithStride from stride -> ATen.newTensorIndexWithSlice from maxBound stride
        SliceUpToWithStride upTo stride -> ATen.newTensorIndexWithSlice 0 upTo stride
        SliceFromUpToWithStride from upTo stride -> ATen.newTensorIndexWithSlice from upTo stride
