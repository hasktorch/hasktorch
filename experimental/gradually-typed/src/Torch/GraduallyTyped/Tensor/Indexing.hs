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
import Data.Kind (Type)
import Data.Singletons (Demote, SingI, SingKind, SomeSing (..), fromSing, sing, toSing, withSomeSing)
import Data.Singletons.Prelude.List (Reverse, SList, Sing)
import Data.Singletons.TH (genSingletons)
import Foreign (fromBool)
import GHC.TypeLits (Div, Nat, Symbol, type (-))
import Torch.GraduallyTyped.Index (Index (..), DemotedIndex (..))
import Torch.GraduallyTyped.Prelude (IsChecked (..), forgetIsChecked)
import Torch.GraduallyTyped.Shape.Class (PrependDimF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import GHC.TypeLits.Extra (Max)
import Data.Coerce (coerce)

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

type RemoveEllipsis :: [IndexType (Index Nat)] -> [IndexType (Index Nat)]
type family RemoveEllipsis indices where
  RemoveEllipsis '[] = '[]
  RemoveEllipsis ('Ellipsis ': ixs) = RemoveEllipsis ixs
  RemoveEllipsis (ix ': ixs) = ix ': RemoveEllipsis ixs

type IndexDimsImpl :: [IndexType (Index Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family IndexDimsImpl indices dims where
  IndexDimsImpl '[] dims = 'Shape dims
  IndexDimsImpl ('None ': ixs) dims = 'Dim ('Name "*") ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('Ellipsis ': _) '[] = 'Shape '[]
  IndexDimsImpl ('Ellipsis ': ixs) dims = ReverseShape (IndexDimsImpl (Reverse (RemoveEllipsis ixs)) (Reverse dims))
  IndexDimsImpl ('SliceAll ': ixs) (dim ': dims) = dim `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceAt _ ': ixs) ('Dim name _ ': dims) = IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'False ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 0) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'True ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFrom 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFrom _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFrom ('Index from) ': ixs) ('Dim name ('Size size) ': dims) = 'Dim name ('Size (size - from)) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpTo 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpTo _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpTo ('Index upTo) ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size upTo) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpTo 'UncheckedIndex _ ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpTo _ 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpTo _ _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpTo ('Index from) ('Index upTo) ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size (upTo - from)) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStride 'UncheckedIndex _ ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStride _ 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStride _ _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStride ('Index from) ('Index stride) ': ixs) ('Dim name ('Size size) ': dims) = 'Dim name ('Size (Div (size - from) stride)) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpToWithStride 'UncheckedIndex _ ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpToWithStride _ 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpToWithStride _ _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpToWithStride ('Index upTo) ('Index stride) ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size (Div upTo stride)) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride 'UncheckedIndex _ _ ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride _ 'UncheckedIndex _ ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride _ _ 'UncheckedIndex ': ixs) ('Dim name _ ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride _ _ _ ': ixs) ('Dim name 'UncheckedSize ': dims) = 'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromUpToWithStride ('Index from) ('Index upTo) ('Index stride) ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size (Max 1 ((upTo - from) `Div` stride))) `PrependDimF` IndexDimsImpl ixs dims

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
    toTensorIndex = fmap fromIntegral >>>
      {- ORMOLU_DISABLE -}
      \case
        None                                     -> ATen.newTensorIndexWithNone
        Ellipsis                                 -> ATen.newTensorIndexWithEllipsis
        SliceAt at                               -> ATen.newTensorIndexWithInt at
        SliceBool b                              -> ATen.newTensorIndexWithBool (fromBool b)
        -- TODO: ATen.newTensorIndexWithTensor
        SliceAll                                 -> ATen.newTensorIndexWithSlice    0 maxBound      1
        SliceFrom               from             -> ATen.newTensorIndexWithSlice from maxBound      1
        SliceUpTo                    upTo        -> ATen.newTensorIndexWithSlice    0     upTo      1
        SliceFromUpTo           from upTo        -> ATen.newTensorIndexWithSlice from     upTo      1
        SliceFromWithStride     from      stride -> ATen.newTensorIndexWithSlice from maxBound stride
        SliceUpToWithStride          upTo stride -> ATen.newTensorIndexWithSlice    0     upTo stride
        SliceFromUpToWithStride from upTo stride -> ATen.newTensorIndexWithSlice from     upTo stride
      {- ORMOLU_ENABLE -}
