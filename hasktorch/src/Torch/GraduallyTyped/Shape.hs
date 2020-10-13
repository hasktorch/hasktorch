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
import Torch.GraduallyTyped.Prelude (Assert, PrependMaybe)
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
data Dim (name :: Type) (size :: Type) where
  AnyDim ::
    forall name size.
    -- | dimension name and dimension size unknown
    Dim name size
  NamedDim ::
    forall name size.
    name ->
    -- | dimension name known and dimension size unknown
    Dim name size
  SizedDim ::
    forall name size.
    size ->
    -- | dimension name unknown and dimension size known
    Dim name size
  NamedSizedDim ::
    forall name size.
    name ->
    size ->
    -- | dimension name and dimension size known
    Dim name size
  deriving (Show)

class KnownDim (dim :: Dim Symbol Nat) where
  dimVal :: Dim String Integer

instance KnownDim 'AnyDim where
  dimVal = AnyDim

instance
  (KnownSymbol name) =>
  KnownDim ( 'NamedDim name)
  where
  dimVal =
    let name = symbolVal $ Proxy @name
     in NamedDim name

instance
  (KnownNat size) =>
  KnownDim ( 'SizedDim size)
  where
  dimVal =
    let size = natVal $ Proxy @size
     in SizedDim size

instance
  (KnownSymbol name, KnownNat size) =>
  KnownDim ( 'NamedSizedDim name size)
  where
  dimVal =
    let name = symbolVal $ Proxy @name
        size = natVal $ Proxy @size
     in NamedSizedDim name size

type UnifyDimNameErrorMessage dim dim' =
  "The supplied dimensions must be the same,"
    % "but dimensions with different names were found:"
    % ""
    % "    " <> dim <> " and " <> dim' <> "."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type family UnifyDimF (dim :: Dim Symbol Nat) (dim' :: Dim Symbol Nat) :: Dim Symbol Nat where
  UnifyDimF 'AnyDim _ = 'AnyDim
  UnifyDimF _ 'AnyDim = 'AnyDim
  UnifyDimF dim dim = dim
  UnifyDimF ( 'NamedDim name) ( 'NamedDim name') =
    TypeError (UnifyDimNameErrorMessage ( 'NamedDim name) ( 'NamedDim name'))
  UnifyDimF ( 'NamedDim name) ( 'NamedSizedDim name _) = 'NamedDim name
  UnifyDimF ( 'NamedDim name) ( 'NamedSizedDim name' size) =
    TypeError (UnifyDimNameErrorMessage ( 'NamedDim name) ( 'NamedSizedDim name' size))
  UnifyDimF ( 'NamedSizedDim name _) ( 'NamedDim name) = 'NamedDim name
  UnifyDimF ( 'NamedSizedDim name size) ( 'NamedDim name') = 
    TypeError (UnifyDimNameErrorMessage ( 'NamedSizedDim name size) ( 'NamedDim name'))
  UnifyDimF ( 'NamedSizedDim name size) ( 'NamedSizedDim name' size) =
    TypeError (UnifyDimNameErrorMessage ( 'NamedSizedDim name size) ( 'NamedSizedDim name' size))
  UnifyDimF dim dim' =
    TypeError
      ( "The supplied dimensions must be the same,"
          % "but different dimensions were found:"
          % ""
          % "    " <> dim <> " and " <> dim' <> "."
          % ""
      )

type AddDimNameErrorMessage dim dim' =
  "Cannot add the dimensions"
    % ""
    % "    '" <> dim <> "' and '" <> dim' <> "'"
    % ""
    % "because they have different names."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type family AddDimF (dim :: Dim Symbol Nat) (dim' :: Dim Symbol Nat) :: Dim Symbol Nat where
  AddDimF 'AnyDim _ = 'AnyDim
  AddDimF _ 'AnyDim = 'AnyDim
  AddDimF ( 'NamedDim name) ( 'NamedDim name) = 'NamedDim name
  AddDimF ( 'NamedDim name) ( 'NamedDim name') =
    TypeError (AddDimNameErrorMessage ( 'NamedDim name) ( 'NamedDim name'))
  AddDimF ( 'NamedDim name) ( 'NamedSizedDim name _) = 'NamedDim name
  AddDimF ( 'NamedDim name) ( 'NamedSizedDim name' size) =
    TypeError (AddDimNameErrorMessage ( 'NamedDim name) ( 'NamedSizedDim name' size))
  AddDimF ( 'NamedSizedDim name _) ( 'NamedDim name) = 'NamedDim name
  AddDimF ( 'NamedSizedDim name size) ( 'NamedDim name') =
    TypeError (AddDimNameErrorMessage ( 'NamedSizedDim name size) ( 'NamedDim name'))
  AddDimF ( 'NamedDim _) _ = 'AnyDim
  AddDimF _ ( 'NamedDim _) = 'AnyDim
  AddDimF ( 'SizedDim size) ( 'SizedDim size') = 'SizedDim (size + size')
  AddDimF ( 'SizedDim size) ( 'NamedSizedDim _ size') = 'SizedDim (size + size')
  AddDimF ( 'NamedSizedDim _ size) ( 'SizedDim size') = 'SizedDim (size + size')
  AddDimF ( 'NamedSizedDim name size) ( 'NamedSizedDim name size') = 'NamedSizedDim name (size + size')
  AddDimF ( 'NamedSizedDim name size) ( 'NamedSizedDim name' size') =
    TypeError (AddDimNameErrorMessage ( 'NamedSizedDim name size) ( 'NamedSizedDim name' size'))

-- | Data type to access dimensions by name or by index.
data DimBy (name :: Type) (index :: Type) where
  AnyDimBy ::
    forall name index.
    -- | unknown dimension access method
    DimBy name index
  DimByName ::
    forall name index.
    name ->
    -- | access dimension by name
    DimBy name index
  DimByIndex ::
    forall name index.
    index ->
    -- | access dimension by index
    DimBy name index

class KnownDimBy (dimBy :: DimBy Symbol Nat) where
  dimByVal :: DimBy String Integer

instance KnownDimBy 'AnyDimBy where
  dimByVal = AnyDimBy

instance
  (KnownSymbol name) =>
  KnownDimBy ( 'DimByName name)
  where
  dimByVal =
    let name = symbolVal $ Proxy @name
     in DimByName name

instance
  (KnownNat index) =>
  KnownDimBy ( 'DimByIndex index)
  where
  dimByVal =
    let index = natVal $ Proxy @index
     in DimByIndex index

class WithDimByC (isAnyDimBy :: Bool) (dimBy :: DimBy Symbol Nat) (f :: Type) where
  type WithDimByF isAnyDimBy f :: Type
  withDimBy :: (DimBy String Integer -> f) -> WithDimByF isAnyDimBy f

instance WithDimByC 'True dimBy f where
  type WithDimByF 'True f = DimBy String Integer -> f
  withDimBy = id

instance (KnownDimBy dimBy) => WithDimByC 'False dimBy f where
  type WithDimByF 'False f = f
  withDimBy f = f (dimByVal @dimBy)

type family NoAnyDimBy dimBy where
  NoAnyDimBy dimBy = TypeError ("No way to prove that " <> dimBy <> " is AnyDimBy. Please specify.")

type IsAnyDimBy dimBy = Assert (NoAnyDimBy dimBy) (dimBy == 'AnyDimBy)

-- | Data type to represent tensor shapes, that is, lists of dimensions.
data Shape (dims :: Type) where
  -- | A fully unknown shape.
  AnyShape ::
    forall dims.
    Shape dims
  -- | A partially known shape.
  -- The list of dimensions has known length, but may contain 'AnyDim'.
  Shape ::
    forall dims.
    dims ->
    Shape dims

class KnownShape (shape :: Shape [Dim Symbol Nat]) where
  shapeVal :: Shape [Dim String Integer]

instance KnownShape 'AnyShape where
  shapeVal = AnyShape

instance KnownShape ( 'Shape '[]) where
  shapeVal = Shape []

instance
  ( KnownShape ( 'Shape dims),
    KnownDim dim
  ) =>
  KnownShape ( 'Shape (dim ': dims))
  where
  shapeVal = case shapeVal @( 'Shape dims) of
    Shape dims -> Shape $ dimVal @dim : dims

class WithShapeC (isAnyShape :: Bool) (shape :: Shape [Dim Symbol Nat]) (f :: Type) where
  type WithShapeF isAnyShape f :: Type
  withShape :: ([Dim String Integer] -> f) -> WithShapeF isAnyShape f

instance WithShapeC 'True shape f where
  type WithShapeF 'True f = [Dim String Integer] -> f
  withShape = id

instance (KnownShape shape) => WithShapeC 'False shape f where
  type WithShapeF 'False f = f
  withShape f = case shapeVal @shape of Shape shape -> f shape

type family UnifyShapeF (shape :: Shape [Dim Symbol Nat]) (shape' :: Shape [Dim Symbol Nat]) :: Shape [Dim Symbol Nat] where
  UnifyShapeF 'AnyShape 'AnyShape = 'AnyShape
  UnifyShapeF ( 'Shape _) 'AnyShape = 'AnyShape
  UnifyShapeF 'AnyShape ( 'Shape _) = 'AnyShape
  UnifyShapeF ( 'Shape dims) ( 'Shape dims) = 'Shape dims
  UnifyShapeF ( 'Shape dims) ( 'Shape dims') = 'Shape (UnifyDimsF dims dims')

type family UnifyDimsF (dims :: [Dim Symbol Nat]) (dims' :: [Dim Symbol Nat]) :: [Dim Symbol Nat] where
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
-- returns the first dimension matching 'dimBy'
-- or 'Nothing' if nothing can be found.
--
-- >>> :kind! GetDimByImplF 'AnyDimBy ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim])
-- GetDimByImplF 'AnyDimBy ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) :: Maybe
--                                                                                                   (Dim
--                                                                                                      Symbol
--                                                                                                      Nat)
-- = 'Just ('NamedDim "batch")
--
-- >>> :kind! GetDimByImplF ('DimByName "feature") ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim])
-- GetDimByImplF ('DimByName "feature") ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) :: Maybe
--                                                                                                                (Dim
--                                                                                                                   Symbol
--                                                                                                                   Nat)
-- = 'Just ('NamedSizedDim "feature" 20)
type family GetDimByImplF (dimBy :: DimBy Symbol Nat) (shape :: Shape [Dim Symbol Nat]) :: Maybe (Dim Symbol Nat) where
  GetDimByImplF 'AnyDimBy 'AnyShape = 'Nothing
  GetDimByImplF 'AnyDimBy ( 'Shape '[]) = 'Nothing
  GetDimByImplF 'AnyDimBy ( 'Shape (h ': _)) = 'Just h
  GetDimByImplF ( 'DimByName _) 'AnyShape = 'Nothing
  GetDimByImplF ( 'DimByName _) ( 'Shape '[]) = 'Nothing
  GetDimByImplF ( 'DimByName name) ( 'Shape ( 'AnyDim : t)) = GetDimByImplF ( 'DimByName name) ( 'Shape t)
  GetDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedDim name) ': _)) = 'Just ( 'NamedDim name)
  GetDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedDim _) ': t)) = GetDimByImplF ( 'DimByName name) ( 'Shape t)
  GetDimByImplF ( 'DimByName name) ( 'Shape (( 'SizedDim _) ': t)) = GetDimByImplF ( 'DimByName name) ( 'Shape t)
  GetDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedSizedDim name size) ': _)) = 'Just ( 'NamedSizedDim name size)
  GetDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedSizedDim _ size) ': t)) = GetDimByImplF ( 'DimByName name) ( 'Shape t)
  GetDimByImplF ( 'DimByIndex _) 'AnyShape = 'Nothing
  GetDimByImplF ( 'DimByIndex index) ( 'Shape dims) = GetDimByIndexImplF index dims

-- | Given a list of dimensions,
-- returns the dimension in the position 'index'
-- or 'Nothing' if 'index' is out of bounds.
type family GetDimByIndexImplF (index :: Nat) (dims :: [Dim Symbol Nat]) :: Maybe (Dim Symbol Nat) where
  GetDimByIndexImplF 0 (h ': _) = Just h
  GetDimByIndexImplF index (_ ': t) = GetDimByIndexImplF (index - 1) t
  GetDimByIndexImplF _ _ = Nothing

type family GetDimByCheckF (dimBy :: DimBy Symbol Nat) (shape :: Shape [Dim Symbol Nat]) (res :: Maybe (Dim Symbol Nat)) :: Dim Symbol Nat where
  GetDimByCheckF dimBy shape 'Nothing =
    TypeError
      ( "Cannot return the first dimension matching"
          % ""
          % "    '" <> dimBy <> "'"
          % ""
          % "in the shape"
          % ""
          % "    '" <> shape <> "'."
          % ""
      )
  GetDimByCheckF _ _ ( 'Just dim) = dim

type GetDimByF dimBy shape = GetDimByCheckF dimBy shape (GetDimByImplF dimBy shape)

-- | Given a 'shape' and a dimension,
-- returns a list of dimensions where the first dimension matching 'dimBy' is replaced
-- or 'Nothing' if nothing can be found.
--
-- >>> :kind! ReplaceDimByImplF 'AnyDimBy ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) 'AnyDim
-- ReplaceDimByImplF 'AnyDimBy ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) 'AnyDim :: Maybe
--                                                                                                               [Dim
--                                                                                                                  Symbol
--                                                                                                                  Nat]
-- = 'Just '[ 'AnyDim, 'NamedSizedDim "feature" 20, 'AnyDim]
--
-- >>> :kind! ReplaceDimByImplF ('DimByName "feature") ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) ('SizedDim 10)
-- ReplaceDimByImplF ('DimByName "feature") ('Shape '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim]) ('SizedDim 10) :: Maybe [Dim Symbol  Nat]
-- = 'Just '[ 'NamedDim "batch", 'SizedDim 10, 'AnyDim]
type family ReplaceDimByImplF (dimBy :: DimBy Symbol Nat) (shape :: Shape [Dim Symbol Nat]) (dim :: Dim Symbol Nat) :: Maybe [Dim Symbol Nat] where
  ReplaceDimByImplF 'AnyDimBy 'AnyShape _ = 'Nothing
  ReplaceDimByImplF 'AnyDimBy ( 'Shape '[]) _ = 'Nothing
  ReplaceDimByImplF 'AnyDimBy ( 'Shape (_ ': t)) dim = 'Just (dim ': t)
  ReplaceDimByImplF ( 'DimByName _) 'AnyShape _ = 'Nothing
  ReplaceDimByImplF ( 'DimByName _) ( 'Shape '[]) _ = 'Nothing
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape ( 'AnyDim ': t)) dim = PrependMaybe ( 'Just 'AnyDim) (ReplaceDimByImplF ( 'DimByName name) ( 'Shape t) dim)
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedDim name) ': t)) dim = 'Just (dim ': t)
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedDim name') ': t)) dim = PrependMaybe ( 'Just ( 'NamedDim name')) (ReplaceDimByImplF ( 'DimByName name) ( 'Shape t) dim)
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape (( 'SizedDim size) ': t)) dim = PrependMaybe ( 'Just ( 'SizedDim size)) (ReplaceDimByImplF ( 'DimByName name) ( 'Shape t) dim)
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedSizedDim name _) ': t)) dim = 'Just (dim ': t)
  ReplaceDimByImplF ( 'DimByName name) ( 'Shape (( 'NamedSizedDim name' size) ': t)) dim = PrependMaybe ( 'Just ( 'NamedSizedDim name' size)) (ReplaceDimByImplF ( 'DimByName name) ( 'Shape t) dim)
  ReplaceDimByImplF ( 'DimByIndex _) 'AnyShape _ = 'Nothing
  ReplaceDimByImplF ( 'DimByIndex index) ( 'Shape dims) dim = ReplaceDimByIndexImplF index dims dim

type family ReplaceDimByCheckF (dimBy :: DimBy Symbol Nat) (shape :: Shape [Dim Symbol Nat]) (dim :: Dim Symbol Nat) (res :: Maybe [Dim Symbol Nat]) :: Shape [Dim Symbol Nat] where
  ReplaceDimByCheckF dimBy shape dim 'Nothing =
    TypeError
      ( "Cannot replace the first dimension matching"
          % ""
          % "    '" <> dimBy <> "'"
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
  ReplaceDimByCheckF _ _ _ ( 'Just dims) = 'Shape dims

type ReplaceDimByF dimBy shape dim = ReplaceDimByCheckF dimBy shape dim (ReplaceDimByImplF dimBy shape dim)

-- | Given a list of dimensions and a dimension,
-- returns a new list of dimensions where the dimension in the position 'index' is replaced
-- or 'Nothing' if 'index' is out of bounds.
--
-- >>> :kind! ReplaceDimByIndexImplF 1 '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim] ('SizedDim 10)
-- ReplaceDimByIndexImplF 1 '[ 'NamedDim "batch", 'NamedSizedDim "feature" 20, 'AnyDim] ('SizedDim 10) :: Maybe
--                                                                                                          [Dim
--                                                                                                             Symbol
--                                                                                                             Nat]
-- = 'Just '[ 'NamedDim "batch", 'SizedDim 10, 'AnyDim]
type family ReplaceDimByIndexImplF (index :: Nat) (dims :: [Dim Symbol Nat]) (dim :: Dim Symbol Nat) :: Maybe [Dim Symbol Nat] where
  ReplaceDimByIndexImplF 0 (_ ': t) dim = Just (dim ': t)
  ReplaceDimByIndexImplF index (h ': t) dim = PrependMaybe ( 'Just h) (ReplaceDimByIndexImplF (index - 1) t dim)
  ReplaceDimByIndexImplF _ _ _ = Nothing

namedDims :: forall m name size. MonadPlus m => [Dim name size] -> m [name]
namedDims = foldM step mempty
  where
    step _ AnyDim = mzero
    step acc (NamedDim name) = pure $ name : acc
    step _ (SizedDim _) = mzero
    step acc (NamedSizedDim name _) = pure $ name : acc

sizedDims :: forall m name size. MonadPlus m => [Dim name size] -> m [size]
sizedDims = foldM step mempty
  where
    step _ AnyDim = mzero
    step _ (NamedDim _) = mzero
    step acc (SizedDim size) = pure $ size : acc
    step acc (NamedSizedDim _ size) = pure $ size : acc

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
