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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Shape where

import Control.Monad (MonadPlus, foldM, mzero)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Type.Equality (type (==))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, Symbol, TypeError, natVal, symbolVal, type (+), type (-))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Assert, Concat, PrependMaybe)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Dimname as ATen (dimname_symbol, fromSymbol_s)
import qualified Torch.Internal.Managed.Type.DimnameList as ATen (dimnameList_at_s, dimnameList_push_back_n, dimnameList_size, newDimnameList)
import qualified Torch.Internal.Managed.Type.IntArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen (newStdString_s, string_c_str)
import qualified Torch.Internal.Managed.Type.Symbol as ATen (dimname_s)
import Torch.Internal.Type (IntArray)
import qualified Torch.Internal.Type as ATen (Dimname, DimnameList)
import Type.Errors.Pretty (type (%), type (<>))

-- | Data type to represent a tensor dimension.
data DimType (name :: Type) (size :: Type) where
  Named ::
    forall name size.
    name ->
    -- | Dimension name is known and dimension size is unknown.
    DimType name size
  Sized ::
    forall name size.
    size ->
    -- | Dimension name is unknown and dimension size is known.
    DimType name size
  NamedSized ::
    forall name size.
    name ->
    size ->
    -- | Both dimension name and dimension size are known.
    DimType name size
  deriving (Show)

class KnownDimType (dimType :: DimType Symbol Nat) where
  dimTypeVal :: DimType String Integer

instance
  (KnownSymbol name) =>
  KnownDimType ( 'Named name)
  where
  dimTypeVal =
    let name = symbolVal $ Proxy @name
     in Named name

instance
  (KnownNat size) =>
  KnownDimType ( 'Sized size)
  where
  dimTypeVal =
    let size = natVal $ Proxy @size
     in Sized size

instance
  (KnownSymbol name, KnownNat size) =>
  KnownDimType ( 'NamedSized name size)
  where
  dimTypeVal =
    let name = symbolVal $ Proxy @name
        size = natVal $ Proxy @size
     in NamedSized name size

-- | Data type to represent whether or not a tensor dimension is checked, that is, known to the compiler.
data Dim (dimType :: Type) where
  -- | The dimension is unchecked, that is, unknown to the compiler.
  UncheckedDim :: forall dimType. Dim dimType
  -- | The dimension is checked, that is, known to the compiler.
  Dim :: forall dimType. dimType -> Dim dimType
  deriving (Show)

class KnownDim (dim :: Dim (DimType Symbol Nat)) where
  dimVal :: Dim (DimType String Integer)

instance KnownDim 'UncheckedDim where
  dimVal = UncheckedDim

instance
  (KnownDimType dimType) =>
  KnownDim ( 'Dim dimType)
  where
  dimVal = Dim (dimTypeVal @dimType)

type UnifyDimNameErrorMessage dim dim' =
  "The supplied dimensions must be the same,"
    % "but dimensions with different names were found:"
    % ""
    % "    " <> dim <> " and " <> dim' <> "."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type UnifyDimSizeErrorMessage dim dim' =
  "The supplied dimensions must be the same,"
    % "but dimensions with different sizes were found:"
    % ""
    % "    " <> dim <> " and " <> dim' <> "."
    % ""
    % "Check whether or not this is really what you want."
    % "If you are certain, adjust the sizes such that they match."

type UnifyDimNameSizeErrorMessage dim dim' =
  "The supplied dimensions must be the same,"
    % "but very different dimensions were found:"
    % ""
    % "    " <> dim <> " and " <> dim' <> "."
    % ""
    % "Both names and sizes disagree."
    % "It's very unlikely that this is what you want."

type family UnifyDimTypeF (dimType :: DimType Symbol Nat) (dimType' :: DimType Symbol Nat) :: DimType Symbol Nat where
  UnifyDimTypeF dimType dimType = dimType
  UnifyDimTypeF ( 'Named name) ( 'Named name') = TypeError (UnifyDimNameErrorMessage ( 'Named name) ( 'Named name'))
  UnifyDimTypeF ( 'Named name) ( 'Sized _) = 'Named name
  UnifyDimTypeF ( 'Named name) ( 'NamedSized name _) = 'Named name
  UnifyDimTypeF ( 'Named name) ( 'NamedSized name' size) = TypeError (UnifyDimNameErrorMessage ( 'Named name) ( 'NamedSized name' size))
  UnifyDimTypeF ( 'Sized _) ( 'Named name) = 'Named name
  UnifyDimTypeF ( 'Sized size) ( 'Sized size') = TypeError (UnifyDimSizeErrorMessage ( 'Sized size) ( 'Sized size'))
  UnifyDimTypeF ( 'Sized size) ( 'NamedSized name size) = 'NamedSized name size
  UnifyDimTypeF ( 'Sized size) ( 'NamedSized name size') = TypeError (UnifyDimSizeErrorMessage ( 'Sized size) ( 'NamedSized name size'))
  UnifyDimTypeF ( 'NamedSized name _) ( 'Named name) = 'Named name
  UnifyDimTypeF ( 'NamedSized name size) ( 'Named name') = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'Named name'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'Sized size) = 'NamedSized name size
  UnifyDimTypeF ( 'NamedSized name size) ( 'Sized size') = TypeError (UnifyDimSizeErrorMessage ( 'NamedSized name size) ( 'Sized size'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size) = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name size') = TypeError (UnifyDimSizeErrorMessage ( 'NamedSized name size) ( 'NamedSized name size'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size') = TypeError (UnifyDimNameSizeErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size'))

-- | Unification of dimensions.
--
-- The unification rules are the same as for PyTorch, see
-- https://pytorch.org/docs/stable/named_tensor.html#match-semantics.
type family UnifyDimF (dim :: Dim (DimType Symbol Nat)) (dim' :: Dim (DimType Symbol Nat)) :: Dim (DimType Symbol Nat) where
  UnifyDimF 'UncheckedDim _ = 'UncheckedDim
  UnifyDimF _ 'UncheckedDim = 'UncheckedDim
  UnifyDimF ( 'Dim dimType) ( 'Dim dimType') = 'Dim (UnifyDimTypeF dimType dimType')

type AddDimNameErrorMessage dim dim' =
  "Cannot add the dimensions"
    % ""
    % "    '" <> dim <> "' and '" <> dim' <> "'"
    % ""
    % "because they have different names."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type family AddDimTypeF (dimType :: DimType Symbol Nat) (dimType' :: DimType Symbol Nat) :: DimType Symbol Nat where
  AddDimTypeF ( 'Named name) ( 'Named name) = 'Named name
  AddDimTypeF ( 'Named name) ( 'Named name') = TypeError (AddDimNameErrorMessage ( 'Named name) ( 'Named name'))
  AddDimTypeF ( 'Named name) ( 'Sized _) = 'Named name
  AddDimTypeF ( 'Named name) ( 'NamedSized name _) = 'Named name
  AddDimTypeF ( 'Named name) ( 'NamedSized name' size) = TypeError (AddDimNameErrorMessage ( 'Named name) ( 'NamedSized name' size))
  AddDimTypeF ( 'Sized _) ( 'Named name) = 'Named name
  AddDimTypeF ( 'Sized size) ( 'Sized size') = 'Sized (size + size')
  AddDimTypeF ( 'Sized size) ( 'NamedSized name size') = 'NamedSized name (size + size')
  AddDimTypeF ( 'NamedSized name _) ( 'Named name) = 'Named name
  AddDimTypeF ( 'NamedSized name size) ( 'Named name') = TypeError (AddDimNameErrorMessage ( 'NamedSized name size) ( 'Named name'))
  AddDimTypeF ( 'NamedSized name size) ( 'Sized size') = 'NamedSized name (size + size')
  AddDimTypeF ( 'NamedSized name size) ( 'NamedSized name size') = 'NamedSized name (size + size')
  AddDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size') = TypeError (AddDimNameErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size'))

-- | Addition of dimensions.
--
-- The unification rules are the same as for PyTorch, see
-- https://pytorch.org/docs/stable/named_tensor.html#match-semantics.
type family AddDimF (dim :: Dim (DimType Symbol Nat)) (dim' :: Dim (DimType Symbol Nat)) :: Dim (DimType Symbol Nat) where
  AddDimF 'UncheckedDim _ = 'UncheckedDim
  AddDimF _ 'UncheckedDim = 'UncheckedDim
  AddDimF ( 'Dim dimType) ( 'Dim dimType') = 'Dim (AddDimTypeF dimType dimType')

-- | Data type to select dimensions by name or by index.
data By (name :: Type) (index :: Type) where
  ByName ::
    forall name index.
    name ->
    By name index
  -- | Select a dimension by index.
  ByIndex ::
    forall name index.
    index ->
    By name index
  deriving (Show)

class KnownBy (by :: By Symbol Nat) where
  byVal :: By String Integer

instance
  (KnownSymbol name) =>
  KnownBy ( 'ByName name)
  where
  byVal =
    let name = symbolVal $ Proxy @name
     in ByName name

instance
  (KnownNat index) =>
  KnownBy ( 'ByIndex index)
  where
  byVal =
    let index = natVal $ Proxy @index
     in ByIndex index

data SelectDim (by :: Type) where
  -- Unknown method of dimension selection.
  UncheckedSelectDim :: forall by. SelectDim by
  -- Known method of dimension selection, that is, either by name or by index.
  SelectDim :: forall by. by -> SelectDim by

class KnownSelectDim (selectDim :: SelectDim (By Symbol Nat)) where
  selectDimVal :: SelectDim (By String Integer)

instance KnownSelectDim 'UncheckedSelectDim where
  selectDimVal = UncheckedSelectDim

instance (KnownBy by) => KnownSelectDim ( 'SelectDim by) where
  selectDimVal = let by = byVal @by in SelectDim by

class WithSelectDimC (isUncheckedSelectDim :: Bool) (selectDim :: SelectDim (By Symbol Nat)) (f :: Type) where
  type WithSelectDimF isUncheckedSelectDim f :: Type
  withSelectDim :: (By String Integer -> f) -> WithSelectDimF isUncheckedSelectDim f

instance WithSelectDimC 'True selectDim f where
  type WithSelectDimF 'True f = By String Integer -> f
  withSelectDim = id

instance (KnownSelectDim selectDim) => WithSelectDimC 'False selectDim f where
  type WithSelectDimF 'False f = f
  withSelectDim f = case selectDimVal @selectDim of SelectDim by -> f by

type family NoUncheckedSelectDim selectDim where
  NoUncheckedSelectDim selectDim = TypeError ("No way to prove that " <> selectDim <> " is UncheckedSelectDim. Please specify.")

type IsUncheckedSelectDimF selectDim = Assert (NoUncheckedSelectDim selectDim) (selectDim == 'UncheckedSelectDim)

-- | Data type to represent tensor shapes, that is, lists of dimensions.
data Shape (dims :: Type) where
  -- | The shape is fully unchecked.
  -- Neither the number of the dimensions
  -- nor any dimension properties are known to the compiler.
  UncheckedShape ::
    forall dims.
    Shape dims
  -- | The shape is partially known to the compiler.
  -- The list of dimensions has a known length, but may contain 'UncheckedDim', that is, unknown dimensions.
  Shape ::
    forall dims.
    dims ->
    Shape dims
  deriving (Show)

class KnownShape k (shape :: Shape [k]) where
  type DimValF k :: Type
  shapeVal :: Shape [DimValF k]

instance KnownShape (DimType Symbol Nat) 'UncheckedShape where
  type DimValF (DimType Symbol Nat) = DimType String Integer
  shapeVal = UncheckedShape

instance KnownShape (DimType Symbol Nat) ( 'Shape '[]) where
  type DimValF (DimType Symbol Nat) = DimType String Integer
  shapeVal = Shape []

instance
  ( KnownShape (DimType Symbol Nat) ( 'Shape dimTypes),
    KnownDimType dimType
  ) =>
  KnownShape (DimType Symbol Nat) ( 'Shape (dimType ': dimTypes))
  where
  type DimValF (DimType Symbol Nat) = DimType String Integer
  shapeVal =
    case shapeVal @_ @( 'Shape dimTypes) of
      Shape dimTypes -> Shape $ dimTypeVal @dimType : dimTypes

instance KnownShape (Dim (DimType Symbol Nat)) 'UncheckedShape where
  type DimValF (Dim (DimType Symbol Nat)) = Dim (DimType String Integer)
  shapeVal = UncheckedShape

instance KnownShape (Dim (DimType Symbol Nat)) ( 'Shape '[]) where
  type DimValF (Dim (DimType Symbol Nat)) = Dim (DimType String Integer)
  shapeVal = Shape []

instance
  ( KnownShape (Dim (DimType Symbol Nat)) ( 'Shape dims),
    KnownDim dim
  ) =>
  KnownShape (Dim (DimType Symbol Nat)) ( 'Shape (dim ': dims))
  where
  type DimValF (Dim (DimType Symbol Nat)) = Dim (DimType String Integer)
  shapeVal =
    case shapeVal @_ @( 'Shape dims) of
      Shape dims -> Shape $ dimVal @dim : dims

class WithShapeC (isUncheckedShape :: Bool) (shape :: Shape [DimType Symbol Nat]) (f :: Type) where
  type WithShapeF isUncheckedShape f :: Type
  withShape :: ([DimType String Integer] -> f) -> WithShapeF isUncheckedShape f

instance WithShapeC 'True shape f where
  type WithShapeF 'True f = [DimType String Integer] -> f
  withShape = id

instance (KnownShape (DimType Symbol Nat) shape) => WithShapeC 'False shape f where
  type WithShapeF 'False f = f
  withShape f = case shapeVal @_ @shape of Shape shape -> f shape

type family ConcatShapesF (shape :: Shape [k]) (shape' :: Shape [k]) :: Shape [k] where
  ConcatShapesF 'UncheckedShape _ = 'UncheckedShape
  ConcatShapesF _ 'UncheckedShape = 'UncheckedShape
  ConcatShapesF ( 'Shape dims) ( 'Shape dims') = 'Shape (Concat dims dims')

type family WidenShapeF (shape :: Shape [DimType Symbol Nat]) :: Shape [Dim (DimType Symbol Nat)] where
  WidenShapeF 'UncheckedShape = 'UncheckedShape
  WidenShapeF ( 'Shape '[]) = 'Shape '[]
  WidenShapeF ( 'Shape (dimType ': dimTypes)) = ConcatShapesF ( 'Shape '[ 'Dim dimType]) (WidenShapeF ( 'Shape dimTypes))

type family UnifyShapeF (shape :: Shape [Dim (DimType Symbol Nat)]) (shape' :: Shape [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  UnifyShapeF 'UncheckedShape 'UncheckedShape = 'UncheckedShape
  UnifyShapeF ( 'Shape _) 'UncheckedShape = 'UncheckedShape
  UnifyShapeF 'UncheckedShape ( 'Shape _) = 'UncheckedShape
  UnifyShapeF ( 'Shape dims) ( 'Shape dims) = 'Shape dims
  UnifyShapeF ( 'Shape dims) ( 'Shape dims') = 'Shape (UnifyDimsF dims dims')

type family UnifyDimsF (dims :: [Dim (DimType Symbol Nat)]) (dims' :: [Dim (DimType Symbol Nat)]) :: [Dim (DimType Symbol Nat)] where
  UnifyDimsF '[] '[] = '[]
  UnifyDimsF (dim ': dims) (dim' ': dims') = UnifyDimF dim dim' ': UnifyDimsF dims dims'
  UnifyDimsF _ _ =
    TypeError
      ( "The supplied tensors must have shapes with identical number of dimensions,"
          % "but dimension lists of different lengths were found."
          % ""
          % "Try extending or broadcasting the tensor(s)."
      )

-- | Given a shape,
-- returns the first dimension matching 'selectDim'
-- or 'Nothing' if nothing can be found.
--
-- >>> :kind! GetDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim])
-- GetDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) :: Maybe
--                                                                                                   (Dim
--                                                                                                      Symbol
--                                                                                                      Nat)
-- = 'Just ('Dim ( 'Named "batch"))
--
-- >>> :kind! GetDimImplF ('SelectDim ( 'ByName "feature")) ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim])
-- GetDimImplF ('SelectDim ( 'ByName "feature")) ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) :: Maybe
--                                                                                                                (Dim
--                                                                                                                   Symbol
--                                                                                                                   Nat)
-- = 'Just ('Dim ( 'NamedSized "feature" 20))
type family GetDimImplF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimImplF 'UncheckedSelectDim 'UncheckedShape = 'Nothing
  GetDimImplF 'UncheckedSelectDim ( 'Shape '[]) = 'Nothing
  GetDimImplF 'UncheckedSelectDim ( 'Shape (h ': _)) = 'Just h
  GetDimImplF ( 'SelectDim (ByName _)) 'UncheckedShape = 'Nothing
  GetDimImplF ( 'SelectDim (ByName _)) ( 'Shape '[]) = 'Nothing
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape ( 'UncheckedDim : t)) = GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape t)
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Named name)) ': _)) = 'Just ( 'Dim ( 'Named name))
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Named _)) ': t)) = GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape t)
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Sized _)) ': t)) = GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape t)
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'NamedSized name size)) ': _)) = 'Just ( 'Dim ( 'NamedSized name size))
  GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'NamedSized _ size)) ': t)) = GetDimImplF ( 'SelectDim (ByName name)) ( 'Shape t)
  GetDimImplF ( 'SelectDim (ByIndex _)) 'UncheckedShape = 'Nothing
  GetDimImplF ( 'SelectDim (ByIndex index)) ( 'Shape dims) = GetDimIndexImplF index dims

-- | Given a list of dimensions,
-- returns the dimension in the position 'index'
-- or 'Nothing' if 'index' is out of bounds.
type family GetDimIndexImplF (index :: Nat) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimIndexImplF 0 (h ': _) = Just h
  GetDimIndexImplF index (_ ': t) = GetDimIndexImplF (index - 1) t
  GetDimIndexImplF _ _ = Nothing

type family GetDimCheckF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) (res :: Maybe (Dim (DimType Symbol Nat))) :: Dim (DimType Symbol Nat) where
  GetDimCheckF selectDim shape 'Nothing =
    TypeError
      ( "Cannot return the first dimension matching"
          % ""
          % "    '" <> selectDim <> "'"
          % ""
          % "in the shape"
          % ""
          % "    '" <> shape <> "'."
          % ""
      )
  GetDimCheckF _ _ ( 'Just dim) = dim

type GetDimF selectDim shape = GetDimCheckF selectDim shape (GetDimImplF selectDim shape)

-- | Given a 'shape' and a dimension,
-- returns a list of dimensions where the first dimension matching 'selectDim' is replaced
-- or 'Nothing' if nothing can be found.
--
-- >>> :kind! ReplaceDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) 'UncheckedDim
-- ReplaceDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) 'UncheckedDim :: Maybe
--                                                                                                               [Dim
--                                                                                                                  Symbol
--                                                                                                                  Nat]
-- = 'Just '[ 'UncheckedDim, 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]
--
-- >>> :kind! ReplaceDimImplF ('SelectDim ( 'ByName "feature")) ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) ('SizedDim 10)
-- ReplaceDimImplF ('SelectDim ( 'ByName "feature")) ('Shape '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim]) ('SizedDim 10) :: Maybe
--                                                                                                                                   [Dim
--                                                                                                                                      Symbol
--                                                                                                                                      Nat]
-- = 'Just '[ 'Dim ( 'Named "batch"), 'SizedDim 10, 'UncheckedDim]
type family ReplaceDimImplF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimImplF 'UncheckedSelectDim 'UncheckedShape _ = 'Nothing
  ReplaceDimImplF 'UncheckedSelectDim ( 'Shape '[]) _ = 'Nothing
  ReplaceDimImplF 'UncheckedSelectDim ( 'Shape (_ ': t)) dim = 'Just (dim ': t)
  ReplaceDimImplF ( 'SelectDim (ByName _)) 'UncheckedShape _ = 'Nothing
  ReplaceDimImplF ( 'SelectDim (ByName _)) ( 'Shape '[]) _ = 'Nothing
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape ( 'UncheckedDim ': t)) dim = PrependMaybe ( 'Just 'UncheckedDim) (ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape t) dim)
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Named name)) ': t)) dim = 'Just (dim ': t)
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Named name')) ': t)) dim = PrependMaybe ( 'Just ( 'Dim ( 'Named name'))) (ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape t) dim)
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'Sized size)) ': t)) dim = PrependMaybe ( 'Just ( 'Dim ( 'Sized size))) (ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape t) dim)
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'NamedSized name _)) ': t)) dim = 'Just (dim ': t)
  ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape (( 'Dim ( 'NamedSized name' size)) ': t)) dim = PrependMaybe ( 'Just ( 'Dim ( 'NamedSized name' size))) (ReplaceDimImplF ( 'SelectDim (ByName name)) ( 'Shape t) dim)
  ReplaceDimImplF ( 'SelectDim (ByIndex _)) 'UncheckedShape _ = 'Nothing
  ReplaceDimImplF ( 'SelectDim (ByIndex index)) ( 'Shape dims) dim = ReplaceDimIndexImplF index dims dim

type family ReplaceDimCheckF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) (res :: Maybe [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  ReplaceDimCheckF selectDim shape dim 'Nothing =
    TypeError
      ( "Cannot replace the first dimension matching"
          % ""
          % "    '" <> selectDim <> "'"
          % ""
          % "in the shape"
          % ""
          % "    '" <> shape <> "'"
          % ""
          % "with"
          % ""
          % "    '" <> dim <> "'."
          % ""
      )
  ReplaceDimCheckF _ _ _ ( 'Just dims) = 'Shape dims

type ReplaceDimF selectDim shape dim = ReplaceDimCheckF selectDim shape dim (ReplaceDimImplF selectDim shape dim)

-- | Given a list of dimensions and a dimension,
-- returns a new list of dimensions where the dimension in the position 'index' is replaced
-- or 'Nothing' if 'index' is out of bounds.
--
-- >>> :kind! ReplaceDimIndexImplF 1 '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim] ('SizedDim 10)
-- ReplaceDimIndexImplF 1 '[ 'Dim ( 'Named "batch"), 'Dim ( 'NamedSized "feature" 20), 'UncheckedDim] ('SizedDim 10) :: Maybe
--                                                                                                          [Dim
--                                                                                                             Symbol
--                                                                                                             Nat]
-- = 'Just '[ 'Dim ( 'Named "batch"), 'SizedDim 10, 'UncheckedDim]
type family ReplaceDimIndexImplF (index :: Nat) (dims :: [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimIndexImplF 0 (_ ': t) dim = Just (dim ': t)
  ReplaceDimIndexImplF index (h ': t) dim = PrependMaybe ( 'Just h) (ReplaceDimIndexImplF (index - 1) t dim)
  ReplaceDimIndexImplF _ _ _ = Nothing

namedDims :: forall m name size. MonadPlus m => [DimType name size] -> m [name]
namedDims dims = reverse <$> foldM step mempty dims
  where
    step acc (Named name) = pure $ name : acc
    step _ (Sized _) = mzero
    step acc (NamedSized name _) = pure $ name : acc

sizedDims :: forall m name size. MonadPlus m => [DimType name size] -> m [size]
sizedDims dims = reverse <$> foldM step mempty dims
  where
    step _ (Named _) = mzero
    step acc (Sized size) = pure $ size : acc
    step acc (NamedSized _ size) = pure $ size : acc

instance Castable String (ForeignPtr ATen.Dimname) where
  cast name f =
    let ptr = unsafePerformIO $ do
          str <- ATen.newStdString_s name
          symbol <- ATen.dimname_s str
          ATen.fromSymbol_s symbol
     in f ptr
  uncast ptr f =
    let name = unsafePerformIO $ do
          symbol <- ATen.dimname_symbol ptr
          str <- undefined symbol
          ATen.string_c_str str
     in f name

instance Castable [ForeignPtr ATen.Dimname] (ForeignPtr ATen.DimnameList) where
  cast names f =
    let ptr = unsafePerformIO $ do
          list <- ATen.newDimnameList
          mapM_ (ATen.dimnameList_push_back_n list) names
          return list
     in f ptr
  uncast ptr f =
    let names = unsafePerformIO $ do
          len <- ATen.dimnameList_size ptr
          mapM (ATen.dimnameList_at_s ptr) [0 .. (len - 1)]
     in f names

instance Castable [String] (ForeignPtr ATen.DimnameList) where
  cast xs f = do
    ptrList <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Dimname))) xs
    cast ptrList f
  uncast xs f = uncast xs $ \ptrList -> do
    names <- mapM (\(x :: ForeignPtr ATen.Dimname) -> uncast x return) $ ptrList
    f names

instance Castable [Integer] (ForeignPtr IntArray) where
  cast sizes f =
    let ptr = unsafePerformIO $ do
          array <- ATen.newIntArray
          mapM_ (ATen.intArray_push_back_l array . fromInteger) sizes
          return array
     in f ptr
  uncast ptr f =
    let sizes = unsafePerformIO $ do
          len <- ATen.intArray_size ptr
          mapM ((<$>) toInteger . ATen.intArray_at_s ptr) [0 .. (len - 1)]
     in f sizes
