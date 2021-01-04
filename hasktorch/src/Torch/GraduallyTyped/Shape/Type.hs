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

module Torch.GraduallyTyped.Shape.Type where

import Control.Monad (MonadPlus, foldM, mzero)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat (..), KnownSymbol (..), Nat, Symbol, natVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Dimname as ATen (dimname_symbol, fromSymbol_s)
import qualified Torch.Internal.Managed.Type.DimnameList as ATen (dimnameList_at_s, dimnameList_push_back_n, dimnameList_size, newDimnameList)
import qualified Torch.Internal.Managed.Type.IntArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen (newStdString_s, string_c_str)
import qualified Torch.Internal.Managed.Type.Symbol as ATen (dimname_s, symbol_toUnqualString)
import Torch.Internal.Type (IntArray)
import qualified Torch.Internal.Type as ATen (Dimname, DimnameList)

data Size (size :: Type) where
  UncheckedSize :: forall size. Size size
  Size :: forall size. size -> Size size
  deriving (Show)

class KnownSize (size :: Size Nat) where
  sizeVal :: Size Integer

instance KnownSize 'UncheckedSize where
  sizeVal = UncheckedSize

instance KnownNat size => KnownSize ( 'Size size) where
  sizeVal = Size (natVal $ Proxy @size)

data Name (name :: Type) where
  UncheckedName :: forall name. Name name
  Name :: forall name. name -> Name name
  deriving (Show)

class KnownName (name :: Name Symbol) where
  nameVal :: Name String

instance KnownName 'UncheckedName where
  nameVal = UncheckedName

instance KnownSymbol name => KnownName ( 'Name name) where
  nameVal = Name (symbolVal $ Proxy @name)

data Dim (name :: Type) (size :: Type) where
  Dim :: forall name size. name -> size -> Dim name size
  deriving (Eq, Ord, Show)

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

checkDim ::
  forall dim.
  KnownDim dim =>
  Dim String Integer ->
  Bool
checkDim (Dim name size) =
  case dimVal @dim of
    Dim UncheckedName UncheckedSize -> True
    Dim (Name name') UncheckedSize -> name == name'
    Dim UncheckedName (Size size') -> size == size'
    Dim (Name name') (Size size') -> name == name' && size == size'

unifyDim ::
  forall m.
  MonadFail m =>
  Dim String Integer ->
  Dim String Integer ->
  m (Dim String Integer)
unifyDim (Dim name size) (Dim name' size') | name == name' && size == size' = pure (Dim name size)
unifyDim (Dim "*" size) (Dim name' size') | size == size' = pure (Dim name' size)
unifyDim (Dim name size) (Dim "*" size') | size == size' = pure (Dim name size)
unifyDim dim dim' =
  fail $
    "The supplied dimensions must be the same,"
      <> "but dimensions with different names and/or sizes were found: "
      <> show dim
      <> " and "
      <> show dim'
      <> "."

unifyDims ::
  forall m.
  MonadFail m =>
  Dim String Integer ->
  [Dim String Integer] ->
  m (Dim String Integer)
unifyDims = foldM unifyDim

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

getDim ::
  forall m.
  MonadFail m =>
  By String Integer ->
  [Dim String Integer] ->
  m (Dim String Integer)
getDim by shape = go 0 shape
  where
    go _ [] = fail $ "Cannot return the first dimension matching " <> show by <> " in the shape " <> show shape <> "."
    go index (dim@(Dim name _) : dims) =
      case by of
        ByName name' | name == name' -> pure dim
        ByIndex index' | index == index' -> pure dim
        _ -> go (index + 1) dims

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
          str <- ATen.symbol_toUnqualString symbol
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
