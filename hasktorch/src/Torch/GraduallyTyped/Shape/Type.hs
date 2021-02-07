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

import Control.Monad (foldM)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat (..), KnownSymbol (..), Nat, Symbol, natVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Concat)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Dimname as ATen (dimname_symbol, fromSymbol_s)
import qualified Torch.Internal.Managed.Type.DimnameList as ATen (dimnameList_at_s, dimnameList_push_back_n, dimnameList_size, newDimnameList)
import qualified Torch.Internal.Managed.Type.IntArray as ATen (intArray_at_s, intArray_push_back_l, intArray_size, newIntArray)
import qualified Torch.Internal.Managed.Type.StdString as ATen (newStdString_s, string_c_str)
import qualified Torch.Internal.Managed.Type.Symbol as ATen (dimname_s, symbol_toUnqualString)
import qualified Torch.Internal.Type as ATen (Dimname, DimnameList, IntArray)

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
  Dim ::
    forall name size.
    { dimName :: name,
      dimSize :: size
    } ->
    Dim name size
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
    "The supplied dimensions must be the same, "
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
  deriving (Show, Eq, Ord)

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

data SelectDims (selectDims :: Type) where
  UncheckedSelectDims ::
    forall selectDims.
    SelectDims selectDims
  SelectDims ::
    forall selectDims.
    selectDims ->
    SelectDims selectDims

class KnownSelectDims (selectDims :: SelectDims [By Symbol Nat]) where
  selectDimsVal :: SelectDims [By String Integer]

instance KnownSelectDims 'UncheckedSelectDims where
  selectDimsVal = UncheckedSelectDims

instance KnownSelectDims ( 'SelectDims '[]) where
  selectDimsVal = SelectDims []

instance
  (KnownBy by, KnownSelectDims ( 'SelectDims bys)) =>
  KnownSelectDims ( 'SelectDims (by ': bys))
  where
  selectDimsVal =
    let by = byVal @by
        SelectDims bys = selectDimsVal @( 'SelectDims bys)
     in SelectDims (by : bys)

class WithSelectDimsC (selectDims :: SelectDims [By Symbol Nat]) (f :: Type) where
  type WithSelectDimsF selectDims f :: Type
  withSelectDims :: ([By String Integer] -> f) -> WithSelectDimsF selectDims f
  withoutSelectDims :: WithSelectDimsF selectDims f -> ([By String Integer] -> f)

instance WithSelectDimsC 'UncheckedSelectDims f where
  type WithSelectDimsF 'UncheckedSelectDims f = [By String Integer] -> f
  withSelectDims = id
  withoutSelectDims = id

instance WithSelectDimsC ( 'SelectDims '[]) f where
  type WithSelectDimsF ( 'SelectDims '[]) f = f
  withSelectDims f = f []
  withoutSelectDims = const

instance
  (WithSelectDimsC ( 'SelectDims selectDims) f, KnownBy by) =>
  WithSelectDimsC ( 'SelectDims (by ': selectDims)) f
  where
  type WithSelectDimsF ( 'SelectDims (by ': selectDims)) f = WithSelectDimsF ( 'SelectDims selectDims) f
  withSelectDims f = withSelectDims @( 'SelectDims selectDims) @f $ \bys -> f (byVal @by : bys)
  withoutSelectDims f (_ : bys) = withoutSelectDims @( 'SelectDims selectDims) @f f bys

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

class
  ShapeConstraint shape (GetShapes f) =>
  WithShapeC (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (f :: Type)
  where
  type WithShapeF shape f :: Type
  withShape :: ([Dim String Integer] -> f) -> WithShapeF shape f
  withoutShape :: WithShapeF shape f -> ([Dim String Integer] -> f)

instance
  ShapeConstraint 'UncheckedShape (GetShapes f) =>
  WithShapeC 'UncheckedShape f
  where
  type WithShapeF 'UncheckedShape f = [Dim String Integer] -> f
  withShape = id
  withoutShape = id

instance
  ( ShapeConstraint ( 'Shape dims) (GetShapes f),
    WithDimsC dims f
  ) =>
  WithShapeC ( 'Shape dims) f
  where
  type WithShapeF ( 'Shape dims) f = WithDimsF dims f
  withShape = withDims @dims
  withoutShape = withoutDims @dims

type family ShapeConstraint (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shapes :: [Shape [Dim (Name Symbol) (Size Nat)]]) :: Constraint where
  ShapeConstraint _ '[] = ()
  ShapeConstraint shape '[shape'] = shape ~ shape'
  ShapeConstraint _ _ = ()

-- >>> :kind! GetShapes ('Shape '[ 'Dim ('Name "*") ('Size 1)])
-- GetShapes ('Shape '[ 'Dim ('Name "*") ('Size 1)]) :: [Shape
--                                                         [Dim (Name Symbol) (Size Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)]]
-- >>> :kind! GetShapes '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)], 'Shape '[ 'Dim 'UncheckedName ('Size 0)]]
-- GetShapes '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)], 'Shape '[ 'Dim 'UncheckedName ('Size 0)]] :: [Shape
--                                                                                                     [Dim
--                                                                                                        (Name
--                                                                                                           Symbol)
--                                                                                                        (Size
--                                                                                                           Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)],
--      'Shape '[ 'Dim 'UncheckedName ('Size 0)]]
-- >>> :kind! GetShapes ('Just ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
-- GetShapes ('Just ('Shape '[ 'Dim ('Name "*") ('Size 1)])) :: [Shape
--                                                                 [Dim (Name Symbol) (Size Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)]]
type family GetShapes (f :: k) :: [Shape [Dim (Name Symbol) (Size Nat)]] where
  GetShapes (a :: Shape [Dim (Name Symbol) (Size Nat)]) = '[a]
  GetShapes (f g) = Concat (GetShapes f) (GetShapes g)
  GetShapes _ = '[]

class WithDimsC (dims :: [Dim (Name Symbol) (Size Nat)]) (f :: Type) where
  type WithDimsF dims f :: Type
  withDims :: ([Dim String Integer] -> f) -> WithDimsF dims f
  withoutDims :: WithDimsF dims f -> ([Dim String Integer] -> f)

instance WithDimsC '[] f where
  type WithDimsF '[] f = f
  withDims f = f []
  withoutDims = const

instance
  WithDimsC dims f =>
  WithDimsC ( 'Dim 'UncheckedName 'UncheckedSize ': dims) f
  where
  type WithDimsF ( 'Dim 'UncheckedName 'UncheckedSize ': dims) f = Dim String Integer -> WithDimsF dims f
  withDims f dim = withDims @dims @f $ \dims -> f (dim : dims)
  withoutDims f (dim : dims) = withoutDims @dims @f (f dim) dims

instance
  ( WithDimsC dims f,
    KnownSymbol name
  ) =>
  WithDimsC ( 'Dim ( 'Name name) 'UncheckedSize ': dims) f
  where
  type WithDimsF ( 'Dim ( 'Name name) 'UncheckedSize ': dims) f = Integer -> WithDimsF dims f
  withDims f size = withDims @dims @f $ \dims -> f (Dim (symbolVal $ Proxy @name) size : dims)
  withoutDims f (Dim _ size : dims) = withoutDims @dims @f (f size) dims

instance
  ( WithDimsC dims f,
    KnownNat size
  ) =>
  WithDimsC ( 'Dim 'UncheckedName ( 'Size size) ': dims) f
  where
  type WithDimsF ( 'Dim 'UncheckedName ( 'Size size) ': dims) f = String -> WithDimsF dims f
  withDims f name = withDims @dims @f $ \dims -> f (Dim name (natVal $ Proxy @size) : dims)
  withoutDims f (Dim name _ : dims) = withoutDims @dims @f (f name) dims

instance
  ( WithDimsC dims f,
    KnownSymbol name,
    KnownNat size
  ) =>
  WithDimsC ( 'Dim ( 'Name name) ( 'Size size) ': dims) f
  where
  type WithDimsF ( 'Dim ( 'Name name) ( 'Size size) ': dims) f = WithDimsF dims f
  withDims f = withDims @dims @f $ \dims -> f (Dim (symbolVal $ Proxy @name) (natVal $ Proxy @size) : dims)
  withoutDims f (_ : dims) = withoutDims @dims @f f dims

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

instance Castable [Integer] (ForeignPtr ATen.IntArray) where
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
