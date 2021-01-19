{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Shape.Class where

import GHC.TypeLits (Symbol, TypeError, type (+), type (-))
import GHC.TypeNats (Nat)
import Torch.GraduallyTyped.Prelude (Fst, LiftTimesMaybe, MapMaybe, PrependMaybe, Reverse, Snd)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import Type.Errors.Pretty (type (%), type (<>))

type family AddSizeF (size :: Size Nat) (size' :: Size Nat) :: Size Nat where
  AddSizeF ( 'Size size) ( 'Size size') = 'Size (size + size')
  AddSizeF size size' = size <+> size'

type family AddDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Dim (Name Symbol) (Size Nat) where
  AddDimF ( 'Dim name size) ( 'Dim name' size') = 'Dim (name <+> name') (AddSizeF size size')

type family BroadcastSizeF (size :: Size Nat) (size' :: Size Nat) :: Maybe (Size Nat) where
  BroadcastSizeF ( 'Size size) ( 'Size size) = 'Just ( 'Size size)
  BroadcastSizeF ( 'Size size) ( 'Size 1) = 'Just ( 'Size size)
  BroadcastSizeF ( 'Size 1) ( 'Size size) = 'Just ( 'Size size)
  BroadcastSizeF ( 'Size _) ( 'Size _) = 'Nothing
  BroadcastSizeF size size' = 'Just (size <+> size')

type family BroadcastDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  BroadcastDimF ( 'Dim name size) ( 'Dim name' size') = MapMaybe ( 'Dim (name <+> name')) (BroadcastSizeF size size')

type family NumelDimF (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe Nat where
  NumelDimF ( 'Dim _ 'UncheckedSize) = 'Nothing
  NumelDimF ( 'Dim _ ( 'Size size)) = 'Just size

type family BroadcastDimsCheckF (dims :: [Dim (Name Symbol) (Size Nat)]) (dims' :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  BroadcastDimsCheckF dims dims' 'Nothing =
    TypeError
      ( "Cannot broadcast the dimensions"
          % ""
          % "    '" <> dims <> "' and '" <> dims' <> "'."
          % ""
          % "You may need to extend, squeeze, or unsqueeze the dimensions manually."
      )
  BroadcastDimsCheckF _ _ ( 'Just dims) = Reverse dims

type family BroadcastDimsImplF (reversedDims :: [Dim (Name Symbol) (Size Nat)]) (reversedDims' :: [Dim (Name Symbol) (Size Nat)]) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  BroadcastDimsImplF '[] reversedDims = 'Just reversedDims
  BroadcastDimsImplF reversedDims '[] = 'Just reversedDims
  BroadcastDimsImplF (dim ': reversedDims) (dim' ': reversedDims') = PrependMaybe (BroadcastDimF dim dim') (BroadcastDimsImplF reversedDims reversedDims')

type BroadcastDimsF dims dims' = BroadcastDimsCheckF dims dims' (BroadcastDimsImplF (Reverse dims) (Reverse dims'))

type family BroadcastShapesF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  BroadcastShapesF shape shape = shape
  BroadcastShapesF ( 'Shape dims) ( 'Shape dims') = 'Shape (BroadcastDimsF dims dims')
  BroadcastShapesF shape shape' = shape <+> shape'

type family NumelDimsF (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  NumelDimsF '[] = 'Just 1
  NumelDimsF (dim ': dims) = LiftTimesMaybe (NumelDimF dim) (NumelDimsF dims)

type family NumelF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  NumelF 'UncheckedShape = 'Nothing
  NumelF ( 'Shape dims) = NumelDimsF dims

type family GetDimAndIndexByNameF (index :: Nat) (result :: (Maybe (Dim (Name Symbol) (Size Nat)), Maybe Nat)) (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: (Maybe (Dim (Name Symbol) (Size Nat)), Maybe Nat) where
  GetDimAndIndexByNameF _ result _ '[] = result
  GetDimAndIndexByNameF index _ name ( 'Dim 'UncheckedName _ ': dims) = GetDimAndIndexByNameF (index + 1) '( 'Just ( 'Dim 'UncheckedName 'UncheckedSize), 'Nothing) name dims
  GetDimAndIndexByNameF index _ name ( 'Dim ( 'Name name) size ': _) = '( 'Just ( 'Dim ( 'Name name) size), 'Just index)
  GetDimAndIndexByNameF index result name ( 'Dim ( 'Name _) _ ': dims) = GetDimAndIndexByNameF (index + 1) result name dims

type family GetDimByNameF (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimByNameF name dims = Fst (GetDimAndIndexByNameF 0 '( 'Nothing, 'Nothing) name dims)

type family GetIndexByNameF (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  GetIndexByNameF name dims = Snd (GetDimAndIndexByNameF 0 '( 'Nothing, 'Nothing) name dims)

type family GetDimByIndexF (index :: Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimByIndexF 0 (h ': _) = 'Just h
  GetDimByIndexF index (_ ': t) = GetDimByIndexF (index - 1) t
  GetDimByIndexF _ _ = 'Nothing

type family GetDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimImplF ( 'ByName name) dims = GetDimByNameF name dims
  GetDimImplF ( 'ByIndex index) dims = GetDimByIndexF index dims

type GetDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot return the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family GetDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe (Dim (Name Symbol) (Size Nat))) :: Dim (Name Symbol) (Size Nat) where
  GetDimCheckF by dims 'Nothing = TypeError (GetDimErrorMessage by dims)
  GetDimCheckF _ _ ( 'Just dim) = dim

type family GetDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Dim (Name Symbol) (Size Nat) where
  GetDimF 'UncheckedSelectDim _ = 'Dim 'UncheckedName 'UncheckedSize
  GetDimF _ 'UncheckedShape = 'Dim 'UncheckedName 'UncheckedSize
  GetDimF ( 'SelectDim by) ( 'Shape dims) = GetDimCheckF by dims (GetDimImplF by dims)

type family (!) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (_k :: k) :: Dim (Name Symbol) (Size Nat) where
  (!) shape (index :: Nat) = GetDimF ( 'SelectDim ( 'ByIndex index)) shape
  (!) shape (name :: Symbol) = GetDimF ( 'SelectDim ( 'ByName name)) shape

type family ReplaceDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimByIndexF ( 'Just 0) (_ ': t) dim = 'Just (dim ': t)
  ReplaceDimByIndexF ( 'Just index) (h ': t) dim = PrependMaybe ( 'Just h) (ReplaceDimByIndexF ( 'Just (index - 1)) t dim)
  ReplaceDimByIndexF _ _ _ = 'Nothing

type family ReplaceDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimImplF ( 'ByName name) dims dim = ReplaceDimByIndexF (GetIndexByNameF name dims) dims dim
  ReplaceDimImplF ( 'ByIndex index) dims dim = ReplaceDimByIndexF ( 'Just index) dims dim

type family ReplaceDimNameByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (name :: Name Symbol) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimNameByIndexF ( 'Just 0) ('Dim _ size ': t) name' = 'Just ('Dim name' size ': t)
  ReplaceDimNameByIndexF ( 'Just index) (h ': t) name' = PrependMaybe ( 'Just h) (ReplaceDimNameByIndexF ( 'Just (index - 1)) t name')
  ReplaceDimNameByIndexF _ _ _ = 'Nothing

type family ReplaceDimNameImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (name' :: Name Symbol) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimNameImplF ( 'ByName name) dims name' = ReplaceDimNameByIndexF (GetIndexByNameF name dims) dims name'
  ReplaceDimNameImplF ( 'ByIndex index) dims name' = ReplaceDimNameByIndexF ( 'Just index) dims name'

type family ReplaceDimSizeByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (size' :: Size Nat) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimSizeByIndexF ( 'Just 0) ('Dim name _ ': t) size' = 'Just ('Dim name size' ': t)
  ReplaceDimSizeByIndexF ( 'Just index) (h ': t) size' = PrependMaybe ( 'Just h) (ReplaceDimSizeByIndexF ( 'Just (index - 1)) t size')
  ReplaceDimSizeByIndexF _ _ _ = 'Nothing

type family ReplaceDimSizeImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (size' :: Size Nat) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimSizeImplF ( 'ByName name) dims size' = ReplaceDimSizeByIndexF (GetIndexByNameF name dims) dims size'
  ReplaceDimSizeImplF ( 'ByIndex index) dims size' = ReplaceDimSizeByIndexF ( 'Just index) dims size'

type ReplaceDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) =
  "Cannot replace the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "with"
    % ""
    % "    '" <> dim <> "'."
    % ""

type family ReplaceDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimCheckF by dims dim 'Nothing = TypeError (ReplaceDimErrorMessage by dims dim)
  ReplaceDimCheckF _ _ _ ( 'Just dims) = dims

type family ReplaceDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Shape [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimF 'UncheckedSelectDim _ _ = 'UncheckedShape
  ReplaceDimF _ 'UncheckedShape _ = 'UncheckedShape
  ReplaceDimF ( 'SelectDim by) ( 'Shape dims) dim = 'Shape (ReplaceDimCheckF by dims dim (ReplaceDimImplF by dims dim))

type family InsertDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  InsertDimByIndexF ( 'Just 0) dims dim = 'Just (dim ': dims)
  InsertDimByIndexF ( 'Just index) (h ': t) dim = PrependMaybe ( 'Just h) (InsertDimByIndexF ( 'Just (index - 1)) t dim)
  InsertDimByIndexF _ _ _ = 'Nothing

type family InsertDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  InsertDimImplF ( 'ByName name) dims dim = InsertDimByIndexF (GetIndexByNameF name dims) dims dim
  InsertDimImplF ( 'ByIndex index) dims dim = InsertDimByIndexF ( 'Just index) dims dim

type InsertDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) =
  "Cannot insert the dimension"
    % ""
    % "    '" <> dim <> "'"
    % ""
    % "before the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family InsertDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  InsertDimCheckF by dims dim 'Nothing = TypeError (InsertDimErrorMessage by dims dim)
  InsertDimCheckF _ _ _ ( 'Just dims) = dims

type family InsertDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Shape [Dim (Name Symbol) (Size Nat)] where
  InsertDimF 'UncheckedSelectDim _ _ = 'UncheckedShape
  InsertDimF _ 'UncheckedShape _ = 'UncheckedShape
  InsertDimF ( 'SelectDim by) ( 'Shape dims) dim = 'Shape (InsertDimCheckF by dims dim (InsertDimImplF by dims dim))
