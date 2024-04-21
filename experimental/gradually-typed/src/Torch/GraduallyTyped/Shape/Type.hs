{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Shape.Type where

import Data.Bifunctor (Bifunctor (..))
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, SomeNat (..), SomeSymbol (..), Symbol, natVal, someNatVal, someSymbolVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..), forgetIsChecked)
import Torch.GraduallyTyped.Prelude.List (SList (..))
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

data SSize (size :: Size Nat) where
  SUncheckedSize :: Integer -> SSize 'UncheckedSize
  SSize :: forall size. KnownNat size => SSize ('Size size)

deriving stock instance Show (SSize (size :: Size Nat))

type instance Sing = SSize

instance KnownNat size => SingI ('Size size) where
  sing = SSize

type family SizeF (size :: Size Nat) :: Nat where
  SizeF ('Size size) = size

instance SingKind (Size Nat) where
  type Demote (Size Nat) = IsChecked Integer
  fromSing (SUncheckedSize size) = Unchecked size
  fromSing (SSize :: Sing size) = Checked . natVal $ Proxy @(SizeF size)
  toSing (Unchecked size) = SomeSing . SUncheckedSize $ size
  toSing (Checked size) = case someNatVal size of
    Just (SomeNat (_ :: Proxy size)) -> SomeSing (SSize @size)

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

data SName (name :: Name Symbol) where
  SUncheckedName :: String -> SName 'UncheckedName
  SName :: forall name. KnownSymbol name => SName ('Name name)

deriving stock instance Show (SName (name :: Name Symbol))

pattern SNoName :: SName ('Name "*")
pattern SNoName = SName

type instance Sing = SName

instance KnownSymbol name => SingI ('Name name) where
  sing = SName

type family NameF (name :: Name Symbol) :: Symbol where
  NameF ('Name name) = name

instance SingKind (Name Symbol) where
  type Demote (Name Symbol) = IsChecked String
  fromSing (SUncheckedName name) = Unchecked name
  fromSing (SName :: Sing name) = Checked . symbolVal $ Proxy @(NameF name)
  toSing (Unchecked name) = SomeSing . SUncheckedName $ name
  toSing (Checked name) = case someSymbolVal name of
    SomeSymbol (_ :: Proxy name) -> SomeSing (SName @name)

class KnownName (name :: Name Symbol) where
  nameVal :: Name String

instance KnownName 'UncheckedName where
  nameVal = UncheckedName

instance KnownSymbol name => KnownName ('Name name) where
  nameVal = Name (symbolVal $ Proxy @name)

data Dim (name :: Type) (size :: Type) where
  Dim ::
    forall name size.
    { dimName :: name,
      dimSize :: size
    } ->
    Dim name size
  deriving (Eq, Ord, Show)

instance Functor (Dim a) where
  fmap g (Dim name size) = Dim name (g size)

instance Bifunctor Dim where
  bimap f g (Dim name size) = Dim (f name) (g size)

data SDim (dim :: Dim (Name Symbol) (Size Nat)) where
  SDim ::
    forall name size.
    { sDimName :: SName name,
      sDimSize :: SSize size
    } ->
    SDim ('Dim name size)

deriving stock instance Show (SDim (dim :: Dim (Name Symbol) (Size Nat)))

type instance Sing = SDim

instance (KnownSymbol name, KnownNat size) => SingI ('Dim ('Name name) ('Size size)) where
  sing = SDim (sing @('Name name)) (sing @('Size size))

instance SingKind (Dim (Name Symbol) (Size Nat)) where
  type Demote (Dim (Name Symbol) (Size Nat)) = Dim (IsChecked String) (IsChecked Integer)
  fromSing (SDim name size) = Dim (fromSing name) (fromSing size)
  toSing (Dim name size) =
    withSomeSing name $ \name' ->
      withSomeSing size $ \size' ->
        SomeSing $ SDim name' size'

pattern (:&:) ::
  forall
    (name :: Name Symbol)
    (size :: Size Nat).
  SName name ->
  SSize size ->
  SDim ('Dim name size)
pattern (:&:) name size = SDim name size

infix 9 :&:

class KnownDim (dim :: Dim (Name Symbol) (Size Nat)) where
  dimVal :: Dim (Name String) (Size Integer)

instance (KnownName name, KnownSize size) => KnownDim ('Dim name size) where
  dimVal = Dim (nameVal @name) (sizeVal @size)

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

data SBy (by :: By Symbol Nat) where
  SByName :: forall name. KnownSymbol name => SBy ('ByName name)
  SByIndex :: forall index. KnownNat index => SBy ('ByIndex index)

deriving stock instance Show (SBy (by :: By Symbol Nat))

type instance Sing = SBy

instance KnownSymbol name => SingI ('ByName name :: By Symbol Nat) where
  sing = SByName @name

instance KnownNat index => SingI ('ByIndex index :: By Symbol Nat) where
  sing = SByIndex @index

type family ByNameF (by :: By Symbol Nat) :: Symbol where
  ByNameF ('ByName name) = name

type family ByIndexF (by :: By Symbol Nat) :: Nat where
  ByIndexF ('ByIndex index) = index

instance SingKind (By Symbol Nat) where
  type Demote (By Symbol Nat) = By String Integer
  fromSing (SByName :: Sing by) = ByName . symbolVal $ Proxy @(ByNameF by)
  fromSing (SByIndex :: Sing by) = ByIndex . natVal $ Proxy @(ByIndexF by)
  toSing (ByName name) = case someSymbolVal name of
    SomeSymbol (_ :: Proxy name) -> SomeSing (SByName @name)
  toSing (ByIndex index) = case someNatVal index of
    Just (SomeNat (_ :: Proxy index)) -> SomeSing (SByIndex @index)

class KnownBy (by :: By Symbol Nat) where
  byVal :: By String Integer

instance
  (KnownSymbol name) =>
  KnownBy ('ByName name)
  where
  byVal =
    let name = symbolVal $ Proxy @name
     in ByName name

instance
  (KnownNat index) =>
  KnownBy ('ByIndex index)
  where
  byVal =
    let index = natVal $ Proxy @index
     in ByIndex index

data SelectDim (by :: Type) where
  -- | Unknown method of dimension selection.
  UncheckedSelectDim :: forall by. SelectDim by
  -- | Known method of dimension selection, that is, either by name or by index.
  SelectDim :: forall by. by -> SelectDim by

data SSelectDim (selectDim :: SelectDim (By Symbol Nat)) where
  SUncheckedSelectDim :: By String Integer -> SSelectDim 'UncheckedSelectDim
  SSelectDim :: forall by. SBy by -> SSelectDim ('SelectDim by)

deriving stock instance Show (SSelectDim (selectDim :: SelectDim (By Symbol Nat)))

type instance Sing = SSelectDim

instance SingI (by :: By Symbol Nat) => SingI ('SelectDim by) where
  sing = SSelectDim $ sing @by

instance SingKind (SelectDim (By Symbol Nat)) where
  type Demote (SelectDim (By Symbol Nat)) = IsChecked (By String Integer)
  fromSing (SUncheckedSelectDim by) = Unchecked by
  fromSing (SSelectDim by) = Checked . fromSing $ by
  toSing (Unchecked by) = SomeSing . SUncheckedSelectDim $ by
  toSing (Checked by) = withSomeSing by $ SomeSing . SSelectDim

class KnownSelectDim (selectDim :: SelectDim (By Symbol Nat)) where
  selectDimVal :: SelectDim (By String Integer)

instance KnownSelectDim 'UncheckedSelectDim where
  selectDimVal = UncheckedSelectDim

instance (KnownBy by) => KnownSelectDim ('SelectDim by) where
  selectDimVal = let by = byVal @by in SelectDim by

data SelectDims (selectDims :: Type) where
  UncheckedSelectDims ::
    forall selectDims.
    SelectDims selectDims
  SelectDims ::
    forall selectDims.
    selectDims ->
    SelectDims selectDims

data SSelectDims (selectDims :: SelectDims [By Symbol Nat]) where
  SUncheckedSelectDims :: [By String Integer] -> SSelectDims 'UncheckedSelectDims
  SSelectDims :: forall bys. SList bys -> SSelectDims ('SelectDims bys)

deriving stock instance Show (SSelectDims (selectDims :: SelectDims [By Symbol Nat]))

type instance Sing = SSelectDims

instance SingI bys => SingI ('SelectDims (bys :: [By Symbol Nat])) where
  sing = SSelectDims (sing @bys)

instance SingKind (SelectDims [By Symbol Nat]) where
  type Demote (SelectDims [By Symbol Nat]) = IsChecked [By String Integer]
  fromSing (SUncheckedSelectDims bys) = Unchecked bys
  fromSing (SSelectDims bys) = Checked . fromSing $ bys
  toSing (Unchecked bys) = SomeSing . SUncheckedSelectDims $ bys
  toSing (Checked bys) = withSomeSing bys $ SomeSing . SSelectDims

class KnownSelectDims (selectDims :: SelectDims [By Symbol Nat]) where
  selectDimsVal :: SelectDims [By String Integer]

instance KnownSelectDims 'UncheckedSelectDims where
  selectDimsVal = UncheckedSelectDims

instance KnownSelectDims ('SelectDims '[]) where
  selectDimsVal = SelectDims []

instance
  (KnownBy by, KnownSelectDims ('SelectDims bys)) =>
  KnownSelectDims ('SelectDims (by ': bys))
  where
  selectDimsVal =
    let by = byVal @by
        SelectDims bys = selectDimsVal @('SelectDims bys)
     in SelectDims (by : bys)

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

data SShape (shape :: Shape [Dim (Name Symbol) (Size Nat)]) where
  SUncheckedShape :: [Dim String Integer] -> SShape 'UncheckedShape
  SShape :: forall dims. SList dims -> SShape ('Shape dims)

deriving stock instance Show (SShape (shape :: Shape [Dim (Name Symbol) (Size Nat)]))

type instance Sing = SShape

instance SingI dims => SingI ('Shape (dims :: [Dim (Name Symbol) (Size Nat)])) where
  sing = SShape $ sing @dims

instance SingKind (Shape [Dim (Name Symbol) (Size Nat)]) where
  type Demote (Shape [Dim (Name Symbol) (Size Nat)]) = IsChecked [Dim (IsChecked String) (IsChecked Integer)]
  fromSing (SUncheckedShape shape) =
    Unchecked
      . fmap (\(Dim name size) -> Dim (Unchecked name) (Unchecked size))
      $ shape
  fromSing (SShape dims) = Checked . fromSing $ dims
  toSing (Unchecked shape) =
    SomeSing . SUncheckedShape
      . fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
      $ shape
  toSing (Checked shape) = withSomeSing shape $ SomeSing . SShape

class KnownShape (shape :: Shape [Dim (Name Symbol) (Size Nat)]) where
  shapeVal :: Shape [Dim (Name String) (Size Integer)]

instance KnownShape 'UncheckedShape where
  shapeVal = UncheckedShape

instance KnownShape ('Shape '[]) where
  shapeVal = Shape []

instance (KnownShape ('Shape dims), KnownDim dim) => KnownShape ('Shape (dim ': dims)) where
  shapeVal =
    case shapeVal @('Shape dims) of
      Shape dims -> Shape $ dimVal @dim : dims

-- >>> :kind! GetShapes ('Shape '[ 'Dim ('Name "*") ('Size 1)])
-- GetShapes ('Shape '[ 'Dim ('Name "*") ('Size 1)]) :: [Shape
--                                                         [Dim (Name Symbol) (Size Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)]]
-- >>> :kind! GetShapes '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)], 'Shape '[ 'Dim UncheckedName ('Size 0)]]
-- GetShapes '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)], 'Shape '[ 'Dim UncheckedName ('Size 0)]] :: [Shape
--                                                                                                     [Dim
--                                                                                                        (Name
--                                                                                                           Symbol)
--                                                                                                        (Size
--                                                                                                           Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)],
--      'Shape '[ 'Dim UncheckedName ('Size 0)]]
-- >>> :kind! GetShapes ('Just ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
-- GetShapes ('Just ('Shape '[ 'Dim ('Name "*") ('Size 1)])) :: [Shape
--                                                                 [Dim (Name Symbol) (Size Nat)]]
-- = '[ 'Shape '[ 'Dim ('Name "*") ('Size 1)]]
type GetShapes :: k -> [Shape [Dim (Name Symbol) (Size Nat)]]
type family GetShapes f where
  GetShapes (a :: Shape [Dim (Name Symbol) (Size Nat)]) = '[a]
  GetShapes (f g) = Concat (GetShapes f) (GetShapes g)
  GetShapes _ = '[]

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
