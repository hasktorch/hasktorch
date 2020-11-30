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
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat (..), KnownSymbol (..), natVal, symbolVal, Nat, Symbol, TypeError, type (+), type (-))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Snd, Fst, LiftTimesMaybe, MapMaybe, PrependMaybe, Reverse)
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

data Size (size :: Type) where
  UncheckedSize :: forall size. Size size
  Size :: forall size. size -> Size size
  deriving (Show)

class KnownSize (size :: Size Nat) where
  sizeVal :: Size Integer

instance KnownSize 'UncheckedSize where
  sizeVal = UncheckedSize

instance KnownNat size => KnownSize ('Size size) where
  sizeVal = Size (natVal $ Proxy @size)

data Name (name :: Type) where
  UncheckedName :: forall name. Name name
  Name :: forall name. name -> Name name
  deriving (Show)

class KnownName (name :: Name Symbol) where
  nameVal :: Name String

instance KnownName 'UncheckedName where
  nameVal = UncheckedName

instance KnownSymbol name => KnownName ('Name name) where
  nameVal = Name (symbolVal $ Proxy @name)

data Dim (name :: Type) (size :: Type) where
  Dim :: forall name size. name -> size -> Dim name size
  deriving (Show)

class KnownDim (dim :: Dim (Name Symbol) (Size Nat)) where
  dimVal :: Dim (Name String) (Size Integer)

instance (KnownName name, KnownSize size) => KnownDim ( 'Dim name size) where
  dimVal = Dim (nameVal @name) (sizeVal @size)

class WithDimC (dim :: Dim (Name Symbol) (Size Nat)) (f :: Type) where
  type WithDimF dim f :: Type
  withDim :: (Dim String Integer -> f) -> WithDimF dim f
  withoutDim :: WithDimF dim f -> (Dim String Integer -> f)

instance WithDimC ( 'Dim 'UncheckedName 'UncheckedSize) f where
  type WithDimF ( 'Dim 'UncheckedName 'UncheckedSize) f = Dim String Integer -> f
  withDim = id
  withoutDim = id

instance (KnownSymbol name) => WithDimC ( 'Dim ( 'Name name) 'UncheckedSize) f where
  type WithDimF ( 'Dim ( 'Name name) 'UncheckedSize) f = Integer -> f
  withDim f size = f (Dim (symbolVal (Proxy @name)) size)
  withoutDim f (Dim _ size) = f size

instance (KnownNat size) => WithDimC ( 'Dim 'UncheckedName ( 'Size size)) f where
  type WithDimF ( 'Dim 'UncheckedName ( 'Size size)) f = String -> f
  withDim f name = f (Dim name (natVal (Proxy @size)))
  withoutDim f (Dim name _) = f name

instance (KnownSymbol name, KnownNat size) => WithDimC ( 'Dim ( 'Name name) ( 'Size size)) f where
  type WithDimF ( 'Dim ( 'Name name) ( 'Size size)) f = f
  withDim f = f (Dim (symbolVal (Proxy @name)) (natVal (Proxy @size)))
  withoutDim = const

type UnifyNameErrorMessage (name :: Symbol) (name' :: Symbol) =
  "The supplied dimensions must be the same,"
    % "but dimensions with different names were found:"
    % ""
    % "    " <> name <> " and " <> name' <> "."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type family UnifyNameF (name :: Name Symbol) (name' :: Name Symbol) :: Name Symbol where
  UnifyNameF 'UncheckedName _ = 'UncheckedName
  UnifyNameF _ 'UncheckedName = 'UncheckedName
  UnifyNameF ( 'Name name) ( 'Name name) = 'Name name
  UnifyNameF ( 'Name name) ( 'Name name') = TypeError (UnifyNameErrorMessage name name')

type UnifySizeErrorMessage (size :: Nat) (size' :: Nat) =
  "The supplied dimensions must be the same,"
    % "but dimensions with different sizes were found:"
    % ""
    % "    " <> size <> " and " <> size' <> "."
    % ""
    % "Check whether or not this is really what you want."
    % "If you are certain, adjust the sizes such that they match."

type family UnifySizeF (size :: Size Nat) (size' :: Size Nat) :: Size Nat where
  UnifySizeF 'UncheckedSize _ = 'UncheckedSize
  UnifySizeF _ 'UncheckedSize = 'UncheckedSize
  UnifySizeF ( 'Size size) ( 'Size size) = 'Size size
  UnifySizeF ( 'Size size) ( 'Size size') = TypeError (UnifySizeErrorMessage size size')

-- | Unification of dimensions.
--
-- The unification rules are the same as for PyTorch, see
-- https://pytorch.org/docs/stable/named_tensor.html#match-semantics.
type family UnifyDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Dim (Name Symbol) (Size Nat) where
  UnifyDimF ( 'Dim name size) ( 'Dim name' size') = 'Dim (UnifyNameF name name') (UnifySizeF size size')

type family AddSizeF (size :: Size Nat) (size' :: Size Nat) :: Size Nat where
  AddSizeF 'UncheckedSize _ = 'UncheckedSize
  AddSizeF _ 'UncheckedSize = 'UncheckedSize
  AddSizeF ( 'Size size) ( 'Size size') = 'Size (size + size')

-- | Addition of dimensions.
--
-- The unification rules are the same as for PyTorch, see
-- https://pytorch.org/docs/stable/named_tensor.html#match-semantics.
type family AddDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Dim (Name Symbol) (Size Nat) where
  AddDimF ( 'Dim name size) ( 'Dim name' size') = 'Dim (UnifyNameF name name') (AddSizeF size size')

type family BroadcastSizeF (size :: Size Nat) (size' :: Size Nat) :: Maybe (Size Nat) where
  BroadcastSizeF 'UncheckedSize _ = 'Just 'UncheckedSize
  BroadcastSizeF _ 'UncheckedSize = 'Just 'UncheckedSize
  BroadcastSizeF ( 'Size size) ( 'Size size) = 'Just ( 'Size size)
  BroadcastSizeF ( 'Size size) ( 'Size 1) = 'Just ( 'Size size)
  BroadcastSizeF ( 'Size 1) ( 'Size size) = 'Just ( 'Size size)
  BroadcastSizeF _ _ = 'Nothing

type family BroadcastDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  BroadcastDimF ( 'Dim name size) ( 'Dim name' size') = MapMaybe ( 'Dim (UnifyNameF name name')) (BroadcastSizeF size size')

type family NumelDimF (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe Nat where
  NumelDimF ( 'Dim _ 'UncheckedSize) = 'Nothing
  NumelDimF ( 'Dim _ ( 'Size size)) = 'Just size

-- | Data type to select dimensions by name or by index.
data By (name :: Type) (index :: Type) where
  -- | Select a dimension by name.
  ByName ::
    forall name index.
    name ->
    By name index
  -- | Select a dimension by index. Counting starts at zero for the first dimension.
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
  -- | Unknown method of dimension selection.
  UncheckedSelectDim :: forall by. SelectDim by
  -- | Known method of dimension selection, that is, either by name or by index.
  SelectDim :: forall by. by -> SelectDim by

class KnownSelectDim (selectDim :: SelectDim (By Symbol Nat)) where
  selectDimVal :: SelectDim (By String Integer)

instance KnownSelectDim 'UncheckedSelectDim where
  selectDimVal = UncheckedSelectDim

instance (KnownBy by) => KnownSelectDim ( 'SelectDim by) where
  selectDimVal = let by = byVal @by in SelectDim by

class WithSelectDimC (selectDim :: SelectDim (By Symbol Nat)) (f :: Type) where
  type WithSelectDimF selectDim f :: Type
  withSelectDim :: (By String Integer -> f) -> WithSelectDimF selectDim f
  withoutSelectDim :: WithSelectDimF selectDim f -> (By String Integer -> f)

instance WithSelectDimC 'UncheckedSelectDim f where
  type WithSelectDimF 'UncheckedSelectDim f = By String Integer -> f
  withSelectDim = id
  withoutSelectDim = id

instance (KnownBy by) => WithSelectDimC ( 'SelectDim by) f where
  type WithSelectDimF ( 'SelectDim by) f = f
  withSelectDim f = f (byVal @by)
  withoutSelectDim = const

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

class KnownShape (shape :: Shape [Dim (Name Symbol) (Size Nat)]) where
  shapeVal :: Shape [Dim (Name String) (Size Integer)]

instance KnownShape 'UncheckedShape where
  shapeVal = UncheckedShape

instance KnownShape ( 'Shape '[]) where
  shapeVal = Shape []

instance (KnownShape ( 'Shape dims), KnownDim dim) => KnownShape ( 'Shape (dim ': dims)) where
  shapeVal =
    case shapeVal @( 'Shape dims) of
      Shape dims -> Shape $ dimVal @dim : dims

class WithShapeC (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (f :: Type) where
  type WithShapeF shape f :: Type
  withShape :: ([Dim String Integer] -> f) -> WithShapeF shape f
  withoutShape :: WithShapeF shape f -> ([Dim String Integer] -> f)

instance WithShapeC 'UncheckedShape f where
  type WithShapeF 'UncheckedShape f = [Dim String Integer] -> f
  withShape = id
  withoutShape = id

instance WithShapeC ( 'Shape '[]) f where
  type WithShapeF ( 'Shape '[]) f = f
  withShape f = f []
  withoutShape = const

instance
  (WithShapeC ( 'Shape dims) f) =>
  WithShapeC ( 'Shape ( 'Dim 'UncheckedName 'UncheckedSize ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'Dim 'UncheckedName 'UncheckedSize ': dims)) f = Dim String Integer -> WithShapeF ( 'Shape dims) f
  withShape f dim = withShape @( 'Shape dims) @f $ \dims -> f (dim : dims)
  withoutShape f (dim : dims) = withoutShape @( 'Shape dims) @f (f dim) dims

instance
  (WithShapeC ( 'Shape dims) f, KnownSymbol name) =>
  WithShapeC ( 'Shape ( 'Dim ( 'Name name) 'UncheckedSize ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'Dim ( 'Name name) 'UncheckedSize ': dims)) f = Integer -> WithShapeF ( 'Shape dims) f
  withShape f size = withShape @( 'Shape dims) @f $ \dims -> f (Dim (symbolVal $ Proxy @name) size : dims)
  withoutShape f (Dim _ size : dims) = withoutShape @( 'Shape dims) @f (f size) dims

instance
  (WithShapeC ( 'Shape dims) f, KnownNat size) =>
  WithShapeC ( 'Shape ( 'Dim 'UncheckedName ( 'Size size) ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'Dim 'UncheckedName ( 'Size size) ': dims)) f = String -> WithShapeF ( 'Shape dims) f
  withShape f name = withShape @( 'Shape dims) @f $ \dims -> f (Dim name (natVal $ Proxy @size) : dims)
  withoutShape f (Dim name _ : dims) = withoutShape @( 'Shape dims) @f (f name) dims

instance
  (WithShapeC ( 'Shape dims) f, KnownSymbol name, KnownNat size) =>
  WithShapeC ( 'Shape ( 'Dim ( 'Name name) ( 'Size size) ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'Dim ( 'Name name) ( 'Size size) ': dims)) f = WithShapeF ( 'Shape dims) f
  withShape f = withShape @( 'Shape dims) @f $ \dims -> f (Dim (symbolVal $ Proxy @name) (natVal $ Proxy @size) : dims)
  withoutShape f (_ : dims) = withoutShape @( 'Shape dims) @f f dims

type UnifyDimsErrorMessage =
  "The supplied tensors must have shapes with identical number of dimensions,"
    % "but dimension lists of different lengths were found."
    % ""
    % "Try extending or broadcasting the tensor(s)."

type family UnifyDimsF (dims :: [Dim (Name Symbol) (Size Nat)]) (dims' :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  UnifyDimsF '[] '[] = '[]
  UnifyDimsF (dim ': dims) (dim' ': dims') = UnifyDimF dim dim' ': UnifyDimsF dims dims'
  UnifyDimsF _ _ = TypeError UnifyDimsErrorMessage

type family UnifyShapeF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  UnifyShapeF 'UncheckedShape _ = 'UncheckedShape
  UnifyShapeF _ 'UncheckedShape = 'UncheckedShape
  UnifyShapeF ( 'Shape dims) ( 'Shape dims') = 'Shape (UnifyDimsF dims dims')

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

type BroadcastDimsF dims dims' = BroadcastDimsCheckF dims dims' (BroadcastDimsImplF dims dims')

type family BroadcastShapesF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  BroadcastShapesF 'UncheckedShape _ = 'UncheckedShape
  BroadcastShapesF _ 'UncheckedShape = 'UncheckedShape
  BroadcastShapesF ( 'Shape dims) ( 'Shape dims') = 'Shape (BroadcastDimsF dims dims')

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

type family ReplaceDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimByIndexF ( 'Just 0) (_ ': t) dim = 'Just (dim ': t)
  ReplaceDimByIndexF ( 'Just index) (h ': t) dim = PrependMaybe ( 'Just h) (ReplaceDimByIndexF ( 'Just (index - 1)) t dim)
  ReplaceDimByIndexF _ _ _ = 'Nothing

type family ReplaceDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimImplF ( 'ByName name) dims dim = ReplaceDimByIndexF (GetIndexByNameF name dims) dims dim
  ReplaceDimImplF ( 'ByIndex index) dims dim = ReplaceDimByIndexF ( 'Just index) dims dim

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
    names <- mapM (\(x :: ForeignPtr ATen.Dimname) -> uncast x return) ptrList
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

dimName :: Dim String Integer -> String
dimName (Dim name _) = name

dimNames :: forall m name size. MonadPlus m => [Dim (Name name) (Size size)] -> m [name]
dimNames dims = reverse <$> foldM step mempty dims
  where
    step acc (Dim (Name name) _) = pure $ name : acc
    step _ (Dim UncheckedName _) = mzero

dimSize :: Dim String Integer -> Integer
dimSize (Dim _ size) = size

dimSizes :: forall m name size. MonadPlus m => [Dim (Name name) (Size size)] -> m [size]
dimSizes dims = reverse <$> foldM step mempty dims
  where
    step acc (Dim _ (Size size)) = pure $ size : acc
    step _ (Dim _ UncheckedSize) = mzero


{-

-- | Data type to represent a tensor dimension.
data DimType (name :: Type) (size :: Type) where
  -- | Dimension name is known and dimension size is unknown.
  Named ::
    forall name size.
    name ->
    DimType name size
  -- | Dimension name is unknown and dimension size is known.
  Sized ::
    forall name size.
    size ->
    DimType name size
  -- | Both dimension name and dimension size are known.
  NamedSized ::
    forall name size.
    name ->
    size ->
    DimType name size
  deriving (Show)

instance (Show name, Eq name, Num size) => Num (DimType name size) where
  Named name + Named name'
    | name == name' = Named name
    | otherwise = error $ "Cannot add dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  Named name + Sized _ = Named name
  Named name + NamedSized name' _
    | name == name' = Named name
    | otherwise = error $ "Cannot add dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  Sized _ + Named name = Named name
  Sized size + Sized size' = Sized (size + size')
  Sized size + NamedSized name size' = NamedSized name (size + size')
  NamedSized name _ + Named name'
    | name == name' = Named name
    | otherwise = error $ "Cannot add dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  NamedSized name size + Sized size' = NamedSized name (size + size')
  NamedSized name size + NamedSized name' size'
    | name == name' = NamedSized name (size + size')
    | otherwise = error $ "Cannot add dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  Named name * Named name'
    | name == name' = Named name
    | otherwise = error $ "Cannot multiply dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  Named name * Sized _ = Named name
  Named name * NamedSized name' _
    | name == name' = Named name
    | otherwise = error $ "Cannot multiply dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  Sized _ * Named name = Named name
  Sized size * Sized size' = Sized (size * size')
  Sized size * NamedSized name size' = NamedSized name (size * size')
  NamedSized name _ * Named name'
    | name == name' = Named name
    | otherwise = error $ "Cannot multiply dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  NamedSized name size * Sized size' = NamedSized name (size * size')
  NamedSized name size * NamedSized name' size'
    | name == name' = NamedSized name (size * size')
    | otherwise = error $ "Cannot multiply dimensions since they have different names, " <> show name <> " and " <> show name' <> "."
  abs (Named name) = Named name
  abs (Sized size) = Sized (abs size)
  abs (NamedSized name size) = NamedSized name (abs size)
  signum (Named name) = Named name
  signum (Sized size) = Sized (signum size)
  signum (NamedSized name size) = NamedSized name (signum size)
  fromInteger n = Sized (fromInteger n)
  negate (Named name) = Named name
  negate (Sized size) = Sized (negate size)
  negate (NamedSized name size) = NamedSized name (negate size)

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

class WithDimC (dim :: Dim (DimType Symbol Nat)) (f :: Type) where
  type WithDimF dim f :: Type
  withDim :: (DimType String Integer -> f) -> WithDimF dim f
  withoutDim :: WithDimF dim f -> (DimType String Integer -> f)

instance WithDimC 'UncheckedDim f where
  type WithDimF 'UncheckedDim f = DimType String Integer -> f
  withDim = id
  withoutDim = id

instance (KnownDimType dimType) => WithDimC ( 'Dim dimType) f where
  type WithDimF ( 'Dim dimType) f = f
  withDim f = f (dimTypeVal @dimType)
  withoutDim = const

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
  UnifyDimTypeF ( 'Named name) ( 'Sized _) = 'Named name -- this is correct because of torch's name propagation rules
  UnifyDimTypeF ( 'Named name) ( 'NamedSized name _) = 'Named name
  UnifyDimTypeF ( 'Named name) ( 'NamedSized name' size) = TypeError (UnifyDimNameErrorMessage ( 'Named name) ( 'NamedSized name' size))
  UnifyDimTypeF ( 'Sized _) ( 'Named name) = 'Named name -- this is correct because of torch's name propagation rules
  UnifyDimTypeF ( 'Sized size) ( 'Sized size') = TypeError (UnifyDimSizeErrorMessage ( 'Sized size) ( 'Sized size'))
  UnifyDimTypeF ( 'Sized size) ( 'NamedSized name size) = 'NamedSized name size -- this is correct because of torch's name propagation rules
  UnifyDimTypeF ( 'Sized size) ( 'NamedSized name size') = TypeError (UnifyDimSizeErrorMessage ( 'Sized size) ( 'NamedSized name size'))
  UnifyDimTypeF ( 'NamedSized name _) ( 'Named name) = 'Named name
  UnifyDimTypeF ( 'NamedSized name size) ( 'Named name') = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'Named name'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'Sized size) = 'NamedSized name size -- this is correct because of torch's name propagation rules
  UnifyDimTypeF ( 'NamedSized name size) ( 'Sized size') = TypeError (UnifyDimSizeErrorMessage ( 'NamedSized name size) ( 'Sized size'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size) = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name size') = TypeError (UnifyDimSizeErrorMessage ( 'NamedSized name size) ( 'NamedSized name size'))
  UnifyDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size') = TypeError (UnifyDimNameSizeErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size'))

type family UnifyDimTypeC (dimType :: DimType Symbol Nat) (dimType' :: DimType Symbol Nat) :: Constraint where
  UnifyDimTypeC dimType dimType' = Catch (UnifyDimTypeF dimType dimType')

-- | Unification of dimensions.
--
-- The unification rules are the same as for PyTorch, see
-- https://pytorch.org/docs/stable/named_tensor.html#match-semantics.
type family UnifyDimF (dim :: Dim (DimType Symbol Nat)) (dim' :: Dim (DimType Symbol Nat)) :: Dim (DimType Symbol Nat) where
  UnifyDimF 'UncheckedDim _ = 'UncheckedDim
  UnifyDimF _ 'UncheckedDim = 'UncheckedDim
  UnifyDimF ( 'Dim dimType) ( 'Dim dimType') = 'Dim (UnifyDimTypeF dimType dimType')

type family UnifyDimC (dim :: Dim (DimType Symbol Nat)) (dim' :: Dim (DimType Symbol Nat)) :: Constraint where
  UnifyDimC dim dim' = Catch (UnifyDimF dim dim')

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
  -- | Unknown method of dimension selection.
  UncheckedSelectDim :: forall by. SelectDim by
  -- | Known method of dimension selection, that is, either by name or by index.
  SelectDim :: forall by. by -> SelectDim by

class KnownSelectDim (selectDim :: SelectDim (By Symbol Nat)) where
  selectDimVal :: SelectDim (By String Integer)

instance KnownSelectDim 'UncheckedSelectDim where
  selectDimVal = UncheckedSelectDim

instance (KnownBy by) => KnownSelectDim ( 'SelectDim by) where
  selectDimVal = let by = byVal @by in SelectDim by

class WithSelectDimC (selectDim :: SelectDim (By Symbol Nat)) (f :: Type) where
  type WithSelectDimF selectDim f :: Type
  withSelectDim :: (By String Integer -> f) -> WithSelectDimF selectDim f
  withoutSelectDim :: WithSelectDimF selectDim f -> (By String Integer -> f)

instance WithSelectDimC 'UncheckedSelectDim f where
  type WithSelectDimF 'UncheckedSelectDim f = By String Integer -> f
  withSelectDim = id
  withoutSelectDim = id

instance (KnownBy by) => WithSelectDimC ( 'SelectDim by) f where
  type WithSelectDimF ( 'SelectDim by) f = f
  withSelectDim f = f (byVal @by)
  withoutSelectDim = const

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

class WithShapeC (shape :: Shape [Dim (DimType Symbol Nat)]) (f :: Type) where
  type WithShapeF shape f :: Type
  withShape :: ([DimType String Integer] -> f) -> WithShapeF shape f
  withoutShape :: WithShapeF shape f -> ([DimType String Integer] -> f)

instance WithShapeC 'UncheckedShape f where
  type WithShapeF 'UncheckedShape f = [DimType String Integer] -> f
  withShape = id
  withoutShape = id

instance {-# OVERLAPPING #-} WithShapeC 'UncheckedShape Void where
  type WithShapeF 'UncheckedShape Void = [DimType String Integer] -> Void
  withShape = undefined
  withoutShape = undefined

instance WithShapeC ( 'Shape '[]) f where
  type WithShapeF ( 'Shape '[]) f = f
  withShape f = f []
  withoutShape = const

instance {-# OVERLAPPING #-} WithShapeC ( 'Shape '[]) Void where
  type WithShapeF ( 'Shape '[]) Void = Void
  withShape = undefined
  withoutShape = undefined

instance
  (WithShapeC ( 'Shape dims) f) =>
  WithShapeC ( 'Shape ( 'UncheckedDim ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'UncheckedDim ': dims)) f = DimType String Integer -> WithShapeF ( 'Shape dims) f
  withShape f dimType = withShape @( 'Shape dims) @f $ \dimTypes -> f (dimType : dimTypes)
  withoutShape f (dimType : dimTypes) = withoutShape @( 'Shape dims) @f (f dimType) dimTypes

instance {-# OVERLAPPING #-} WithShapeC ( 'Shape ( 'UncheckedDim ': dims)) Void where
  type WithShapeF ( 'Shape ( 'UncheckedDim ': dims)) Void = DimType String Integer -> WithShapeF ( 'Shape dims) Void
  withShape = undefined
  withoutShape = undefined

instance
  (WithShapeC ( 'Shape dims) f, KnownDimType dimType) =>
  WithShapeC ( 'Shape ( 'Dim dimType ': dims)) f
  where
  type WithShapeF ( 'Shape ( 'Dim dimType ': dims)) f = WithShapeF ( 'Shape dims) f
  withShape f = withShape @( 'Shape dims) @f $ \dimTypes -> f (dimTypeVal @dimType : dimTypes)
  withoutShape f (_ : dimTypes) = withoutShape @( 'Shape dims) @f f dimTypes

instance {-# OVERLAPPING #-} WithShapeC ( 'Shape ( 'Dim dimType ': dims)) Void where
  type WithShapeF ( 'Shape ( 'Dim dimType ': dims)) Void = WithShapeF ( 'Shape dims) Void
  withShape = undefined
  withoutShape = undefined

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

type family UnifyShapeC (shape :: Shape [Dim (DimType Symbol Nat)]) (shape' :: Shape [Dim (DimType Symbol Nat)]) :: Constraint where
  UnifyShapeC shape shape' = Catch (UnifyShapeF shape shape')

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

type family UnifyDimsC (dims :: [Dim (DimType Symbol Nat)]) (dims' :: [Dim (DimType Symbol Nat)]) :: Constraint where
  UnifyDimsC dims dims' = Catch (UnifyDimsF dims dims')

type family BroadcastShapesF (shape :: Shape [Dim (DimType Symbol Nat)]) (shape' :: Shape [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  BroadcastShapesF 'UncheckedShape 'UncheckedShape = 'UncheckedShape
  BroadcastShapesF 'UncheckedShape ( 'Shape _) = 'UncheckedShape
  BroadcastShapesF ( 'Shape _) 'UncheckedShape = 'UncheckedShape
  BroadcastShapesF ( 'Shape dims) ( 'Shape dims) = 'Shape dims
  BroadcastShapesF ( 'Shape dims) ( 'Shape dims') = 'Shape (BroadcastDimsF dims dims')

type BroadcastDimsF dims dims' = BroadcastDimsCheckF dims dims' (BroadcastDimsImplF (Reverse dims) (Reverse dims'))

type family BroadcastDimsCheckF (dims :: [Dim (DimType Symbol Nat)]) (dims' :: [Dim (DimType Symbol Nat)]) (result :: Maybe [Dim (DimType Symbol Nat)]) :: [Dim (DimType Symbol Nat)] where
  BroadcastDimsCheckF dims dims' 'Nothing =
    TypeError
      ( "Cannot broadcast the dimensions"
          % ""
          % "    '" <> dims <> "' and '" <> dims' <> "'."
          % ""
          % "You may need to extend, squeeze, or unsqueeze the dimensions manually."
      )
  BroadcastDimsCheckF _ _ ( 'Just dims) = Reverse dims

type family BroadcastDimsImplF (reversedDims :: [Dim (DimType Symbol Nat)]) (reversedDims' :: [Dim (DimType Symbol Nat)]) :: Maybe [Dim (DimType Symbol Nat)] where
  BroadcastDimsImplF '[] reversedDims = 'Just reversedDims
  BroadcastDimsImplF reversedDims '[] = 'Just reversedDims
  BroadcastDimsImplF (dim ': reversedDims) (dim' ': reversedDims') = PrependMaybe (BroadcastDimF dim dim') (BroadcastDimsImplF reversedDims reversedDims')

type family BroadcastDimF (dim :: Dim (DimType Symbol Nat)) (dim' :: Dim (DimType Symbol Nat)) :: Maybe (Dim (DimType Symbol Nat)) where
  BroadcastDimF 'UncheckedDim _ = 'Just 'UncheckedDim
  BroadcastDimF _ 'UncheckedDim = 'Just 'UncheckedDim
  BroadcastDimF ( 'Dim dimType) ( 'Dim dimType') = MapMaybe 'Dim (BroadcastDimTypeF dimType dimType')

type family BroadcastDimTypeF (dimType :: DimType Symbol Nat) (dimType' :: DimType Symbol Nat) :: Maybe (DimType Symbol Nat) where
  BroadcastDimTypeF dimType dimType = 'Just dimType
  BroadcastDimTypeF ( 'Named name) ( 'Named name') = TypeError (UnifyDimNameErrorMessage ( 'Named name) ( 'Named name'))
  BroadcastDimTypeF ( 'Named name) ( 'Sized _) = 'Just ( 'Named name) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'Named name) ( 'NamedSized name _) = 'Just ( 'Named name)
  BroadcastDimTypeF ( 'Named name) ( 'NamedSized name' size) = TypeError (UnifyDimNameErrorMessage ( 'Named name) ( 'NamedSized name' size))
  BroadcastDimTypeF ( 'Sized _) ( 'Named name) = 'Just ( 'Named name) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'Sized size) ( 'Sized 1) = 'Just ( 'Sized size)
  BroadcastDimTypeF ( 'Sized 1) ( 'Sized size) = 'Just ( 'Sized size)
  BroadcastDimTypeF ( 'Sized _) ( 'Sized _) = 'Nothing -- broadcasting not possible
  BroadcastDimTypeF ( 'Sized size) ( 'NamedSized name size) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'Sized size) ( 'NamedSized name 1) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'Sized 1) ( 'NamedSized name size) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'Sized _) ( 'NamedSized _ _) = 'Nothing -- broadcasting not possible
  BroadcastDimTypeF ( 'NamedSized name _) ( 'Named name) = 'Just ( 'Named name)
  BroadcastDimTypeF ( 'NamedSized name size) ( 'Named name') = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'Named name'))
  BroadcastDimTypeF ( 'NamedSized name size) ( 'Sized size) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'NamedSized name size) ( 'Sized 1) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'NamedSized name 1) ( 'Sized size) = 'Just ( 'NamedSized name size) -- this is correct because of torch's name propagation rules
  BroadcastDimTypeF ( 'NamedSized _ _) ( 'Sized _) = 'Nothing -- broadcasting not possible
  BroadcastDimTypeF ( 'NamedSized name size) ( 'NamedSized name 1) = 'Just ( 'NamedSized name size)
  BroadcastDimTypeF ( 'NamedSized name 1) ( 'NamedSized name size) = 'Just ( 'NamedSized name size)
  BroadcastDimTypeF ( 'NamedSized name _) ( 'NamedSized name _) = 'Nothing -- broadcasting not possible
  BroadcastDimTypeF ( 'NamedSized name size) ( 'NamedSized name' size') = TypeError (UnifyDimNameErrorMessage ( 'NamedSized name size) ( 'NamedSized name' size'))

type family NumelF (shape :: Shape [Dim (DimType Symbol Nat)]) :: Maybe Nat where
  NumelF 'UncheckedShape = 'Nothing
  NumelF ('Shape dims) = NumelDimsF dims

type family NumelDimsF (dims :: [Dim (DimType Symbol Nat)]) :: Maybe Nat where
  NumelDimsF '[] = 'Just 1
  NumelDimsF (dim ': dims) = LiftTimesMaybe (NumelDimF dim) (NumelDimsF dims)

type family NumelDimF (dim :: Dim (DimType Symbol Nat)) :: Maybe Nat where
  NumelDimF 'UncheckedDim = 'Nothing
  NumelDimF ('Dim dimType) = NumelDimTypeF dimType

type family NumelDimTypeF (dimType :: DimType Symbol Nat) :: Maybe Nat where
  NumelDimTypeF ('Named _) = 'Nothing
  NumelDimTypeF ('Sized size) = 'Just size
  NumelDimTypeF ('NamedSized _ size) = 'Just size

type family CheckSelectDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  CheckSelectDimF _ 'UncheckedShape = 'UncheckedShape
  CheckSelectDimF 'UncheckedSelectDim _ = 'UncheckedShape
  CheckSelectDimF ('SelectDim by) ('Shape dims) = 'Shape (CheckSelectDimDimsImplF by dims)

type family CheckSelectDimDimsImplF (by :: By Symbol Nat) (dims :: [Dim (DimType Symbol Nat)]) :: [Dim (DimType Symbol Nat)] where
  CheckSelectDimDimsImplF ( 'ByName name) dims = CheckSelectDimByNameDimsImplF name dims
  CheckSelectDimDimsImplF ( 'ByIndex index) dims = CheckSelectDimByIndexImplF index dims

type family CheckSelectDimByNameDimsImplF (name :: Symbol) (dims :: [Dim (DimType Symbol Nat)]) :: [Dim (DimType Symbol Nat)] where

type family CheckSelectDimByIndexImplF (index :: Nat) (dims :: [Dim (DimType Symbol Nat)]) :: [Dim (DimType Symbol Nat)] where

-- | Given a shape,
-- returns the first dimension matching 'selectDim'
-- or 'Nothing' if nothing can be found.
--
-- >>> :kind! GetDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim])
-- GetDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) :: Maybe
--                                                                                                                         (Dim
--                                                                                                                            (DimType
--                                                                                                                               Symbol
--                                                                                                                               Nat))
-- = 'Nothing
--
-- >>> :kind! GetDimImplF ('SelectDim ('ByName "feature")) ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim])
-- GetDimImplF ('SelectDim ('ByName "feature")) ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) :: Maybe
--                                                                                                                                      (Dim
--                                                                                                                                         (DimType
--                                                                                                                                            Symbol
--                                                                                                                                            Nat))
-- = 'Just ('Dim ('NamedSized "feature" 20))
type family GetDimImplF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimImplF _ 'UncheckedShape = 'Just 'UncheckedDim
  GetDimImplF 'UncheckedSelectDim _ = 'Just 'UncheckedDim
  GetDimImplF ('SelectDim by) ('Shape dims) = GetDimDimsImplF by dims

type family GetDimDimsImplF (by :: By Symbol Nat) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimDimsImplF ( 'ByName name) dims = GetDimByNameDimsImplF name dims
  GetDimDimsImplF ( 'ByIndex index) dims = GetDimByIndexImplF index dims

type family GetDimByNameDimsImplF (name :: Symbol) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimByNameDimsImplF name dims = SndMaybe (GetDimAndIndexByNameDimsImplF 0 name dims)

type family GetIndexByNameDimsImplF (name :: Symbol) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe Nat where
  GetIndexByNameDimsImplF name dims = FstMaybe (GetDimAndIndexByNameDimsImplF 0 name dims)

type family GetDimAndIndexByNameDimsImplF (index :: Nat) (name :: Symbol) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe (Nat, Dim (DimType Symbol Nat)) where
  GetDimAndIndexByNameDimsImplF _ _ '[] = 'Nothing
  GetDimAndIndexByNameDimsImplF index name ( 'UncheckedDim : t) = GetDimAndIndexByNameDimsImplF (index + 1) name t
  GetDimAndIndexByNameDimsImplF index name ( 'Dim ( 'Named name) ': _) = 'Just '( index, 'Dim ( 'Named name))
  GetDimAndIndexByNameDimsImplF index name ( 'Dim ( 'Named _) ': t) = GetDimAndIndexByNameDimsImplF (index + 1) name t
  GetDimAndIndexByNameDimsImplF index name ( 'Dim ( 'Sized _) ': t) = GetDimAndIndexByNameDimsImplF (index + 1) name t
  GetDimAndIndexByNameDimsImplF index name ( 'Dim ( 'NamedSized name size) ': _) = 'Just '( index, 'Dim ( 'NamedSized name size))
  GetDimAndIndexByNameDimsImplF index name ( 'Dim ( 'NamedSized _ _) ': t) = GetDimAndIndexByNameDimsImplF (index + 1) name t

-- | Given a list of dimensions,
-- returns the dimension in the position 'index'
-- or 'Nothing' if 'index' is out of bounds.
type family GetDimByIndexImplF (index :: Nat) (dims :: [Dim (DimType Symbol Nat)]) :: Maybe (Dim (DimType Symbol Nat)) where
  GetDimByIndexImplF 0 (h ': _) = Just h
  GetDimByIndexImplF index (_ ': t) = GetDimByIndexImplF (index - 1) t
  GetDimByIndexImplF _ _ = Nothing

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
-- >>> :kind! ReplaceDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) 'UncheckedDim
-- ReplaceDimImplF 'UncheckedSelectDim ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) 'UncheckedDim :: Maybe
--                                                                                                                                           [Dim
--                                                                                                                                              (DimType
--                                                                                                                                                 Symbol
--                                                                                                                                                 Nat)]
-- = 'Just
--     '[ 'UncheckedDim, 'Dim ('NamedSized "feature" 20), 'UncheckedDim]
--
-- >>> :kind! ReplaceDimImplF ('SelectDim ('ByName "feature")) ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) ('Dim ('Sized 10))
-- ReplaceDimImplF ('SelectDim ('ByName "feature")) ('Shape '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim]) ('Dim ('Sized 10)) :: Maybe
--                                                                                                                                                             [Dim
--                                                                                                                                                                (DimType
--                                                                                                                                                                   Symbol
--                                                                                                                                                                   Nat)]
-- = 'Just '[ 'Dim ('Named "batch"), 'Dim ('Sized 10), 'UncheckedDim]
type family ReplaceDimImplF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimImplF _ 'UncheckedShape _ = 'Nothing
  ReplaceDimImplF 'UncheckedSelectDim _ _ = 'Nothing
  ReplaceDimImplF ('SelectDim by) ('Shape dims) dim = ReplaceDimDimsImplF by dims dim

type family ReplaceDimDimsImplF (by :: By Symbol Nat) (dims :: [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimDimsImplF ('ByName name) dims dim = ReplaceDimByNameDimsImplF name dims dim
  ReplaceDimDimsImplF ('ByIndex index) dims dim = ReplaceDimByIndexDimsImplF index dims dim

type family ReplaceDimByNameDimsImplF (name :: Symbol) (dims :: [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimByNameDimsImplF _ '[] _ = 'Nothing
  ReplaceDimByNameDimsImplF name ( 'UncheckedDim ': t) dim = PrependMaybe ( 'Just 'UncheckedDim) (ReplaceDimByNameDimsImplF name t dim)
  ReplaceDimByNameDimsImplF name ( ( 'Dim ( 'Named name)) ': t) dim = 'Just (dim ': t)
  ReplaceDimByNameDimsImplF name ( ( 'Dim ( 'Named name')) ': t) dim = PrependMaybe ( 'Just ( 'Dim ( 'Named name'))) (ReplaceDimByNameDimsImplF name t dim)
  ReplaceDimByNameDimsImplF name ( ( 'Dim ( 'Sized size)) ': t) dim = PrependMaybe ( 'Just ( 'Dim ( 'Sized size))) (ReplaceDimByNameDimsImplF name t dim)
  ReplaceDimByNameDimsImplF name ( ( 'Dim ( 'NamedSized name _)) ': t) dim = 'Just (dim ': t)
  ReplaceDimByNameDimsImplF name ( ( 'Dim ( 'NamedSized name' size)) ': t) dim = PrependMaybe ( 'Just ( 'Dim ( 'NamedSized name' size))) (ReplaceDimByNameDimsImplF name t dim)
 
-- | Given a list of dimensions and a dimension,
-- returns a new list of dimensions where the dimension in the position 'index' is replaced
-- or 'Nothing' if 'index' is out of bounds.
--
-- >>> :kind! ReplaceDimIndexImplF 1 '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim] ('Dim ('Sized 10))
-- ReplaceDimIndexImplF 1 '[ 'Dim ('Named "batch"), 'Dim ('NamedSized "feature" 20), 'UncheckedDim] ('Dim ('Sized 10)) :: Maybe
--                                                                                                                          [Dim
--                                                                                                                             (DimType
--                                                                                                                                Symbol
--                                                                                                                                Nat)]
-- = 'Just '[ 'Dim ('Named "batch"), 'Dim ('Sized 10), 'UncheckedDim]
type family ReplaceDimByIndexDimsImplF (index :: Nat) (dims :: [Dim (DimType Symbol Nat)]) (dim :: Dim (DimType Symbol Nat)) :: Maybe [Dim (DimType Symbol Nat)] where
  ReplaceDimByIndexDimsImplF 0 (_ ': t) dim = Just (dim ': t)
  ReplaceDimByIndexDimsImplF index (h ': t) dim = PrependMaybe ( 'Just h) (ReplaceDimByIndexDimsImplF (index - 1) t dim)
  ReplaceDimByIndexDimsImplF _ _ _ = Nothing


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

-}