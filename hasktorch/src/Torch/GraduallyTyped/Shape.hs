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
import Data.Int (Int16)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits
  ( KnownNat,
    KnownSymbol,
    Nat,
    Symbol,
    natVal,
    symbolVal,
  )
import System.IO.Unsafe (unsafePerformIO)
import Torch.Internal.Cast (cast0, cast1)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Dimname as ATen (dimname_symbol, fromSymbol_s)
import qualified Torch.Internal.Managed.Type.DimnameList as ATen (dimnameList_at_s, dimnameList_push_back_n, dimnameList_size, newDimnameList)
import qualified Torch.Internal.Managed.Type.IntArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen (newStdString_s, string_c_str)
import qualified Torch.Internal.Managed.Type.Symbol as ATen (dimname_s)
import Torch.Internal.Type (IntArray)
import qualified Torch.Internal.Type as ATen (Dimname, DimnameList, Tensor, TensorList)

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
  KnownDim ('NamedDim name)
  where
  dimVal =
    let name = symbolVal $ Proxy @name
     in NamedDim name

instance
  (KnownNat size) =>
  KnownDim ('SizedDim size)
  where
  dimVal =
    let size = natVal $ Proxy @size
     in SizedDim size

instance
  (KnownSymbol name, KnownNat size) =>
  KnownDim ('NamedSizedDim name size)
  where
  dimVal =
    let name = symbolVal $ Proxy @name
        size = natVal $ Proxy @size
     in NamedSizedDim name size

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
  KnownDimBy ('DimByName name)
  where
  dimByVal =
    let name = symbolVal $ Proxy @name
     in DimByName name

instance
  (KnownNat index) =>
  KnownDimBy ('DimByIndex index)
  where
  dimByVal =
    let index = natVal $ Proxy @index
     in DimByIndex index

class DimByC (isAnyDimBy :: Bool) (dimBy :: DimBy Symbol Nat) (f :: Type) where
  type DimByF isAnyDimBy f :: Type
  withDim :: (DimBy String Integer -> f) -> DimByF isAnyDimBy f

instance DimByC 'True dimBy f where
  type DimByF 'True f = DimBy String Integer -> f
  withDim = id

instance (KnownDimBy dimBy) => DimByC 'False dimBy f where
  type DimByF 'False f = f
  withDim f = f (dimByVal @dimBy)

-- | Data type to represent tensor shapes, that is, lists of dimensions
data Shape (shapeList :: Type) where
  AnyShape ::
    forall shapeList.
    -- | shape fully unknown
    Shape shapeList
  Shape ::
    forall shapeList.
    shapeList ->
    -- | shape partially known
    Shape shapeList

class KnownShape (shape :: Shape [Dim Symbol Nat]) where
  shapeVal :: Shape [Dim String Integer]

instance KnownShape 'AnyShape where
  shapeVal = AnyShape

instance KnownShape ('Shape '[]) where
  shapeVal = Shape []

instance
  ( KnownShape ('Shape shapeList),
    KnownDim dim
  ) =>
  KnownShape ('Shape (dim ': shapeList))
  where
  shapeVal = case shapeVal @('Shape shapeList) of
    Shape shapeList -> Shape $ dimVal @dim : shapeList

class WithShapeC (isAnyShape :: Bool) (shape :: Shape [Dim Symbol Nat]) (f :: Type) where
  type WithShapeF isAnyShape f :: Type
  withShape :: ([Dim String Integer] -> f) -> WithShapeF isAnyShape f

instance WithShapeC 'True shape f where
  type WithShapeF 'True f = [Dim String Integer] -> f
  withShape = id

instance (KnownShape shape) => WithShapeC 'False shape f where
  type WithShapeF 'False f = f
  withShape f = case shapeVal @shape of Shape shape -> f shape

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
