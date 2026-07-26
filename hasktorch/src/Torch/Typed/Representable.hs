{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | A 'Representable'-style view of 'NamedTensor'.
--
-- A tensor of shape @'[Batch 2, RGB]@ is a function from an index to an
-- element, so it can be read with 'index' and built with 'tabulate'.  The index
-- type ('Log') is an 'HList' of 'Finite's, one per named dimension, so indices
-- are bounds-checked at compile time:
--
-- > Log (NamedTensor '( 'D.CPU, 0) 'D.Float '[Batch 2, RGB]) ~ HList '[Finite 2, Finite 3]
--
-- Unlike @Data.Functor.Rep.Representable@ this is not a class over a
-- @Type -> Type@: a 'NamedTensor' fixes its element type via its @dtype@, so it
-- is not a functor in the element.  The laws are the usual ones:
--
-- > index (tabulate f) i == f i
-- > tabulate (index t)   == t
module Torch.Typed.Representable
  ( Representable (..),
    HasIndex (..),
    ToFinites,
    dimsOf,
    indexList,
    tabulateList,
    allIndices,
  )
where

import Data.Finite (Finite, getFinite, packFinite)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import Torch.HList
import qualified Torch.Tensor as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Auxiliary (natValI)
import Torch.Typed.Tensor

-- | The index type of a named shape: one 'Finite' per dimension, each bounded
-- by that dimension's 'ToNat'.
type family ToFinites (shape :: Shape) :: [Type] where
  ToFinites '[] = '[]
  ToFinites (x ': xs) = Finite (ToNat x) ': ToFinites xs

-- | Shapes whose indices can be converted to and from plain @[Int]@, and whose
-- dimensions can be reified at runtime.
--
-- Defined over the named 'Shape' rather than over @ToNats shape@ so that GHC
-- never has to prove @ToFinites shape ~ ToFinites' (ToNats shape)@.
class HasIndex (shape :: Shape) where
  -- | The runtime dimensions, e.g. @[2,3]@ for @'[Batch 2, RGB]@.
  dims :: Proxy shape -> [Int]

  -- | Erase a typed index to a list of plain @Int@s.
  toInts :: HList (ToFinites shape) -> [Int]

  -- | Reconstruct a typed index.  The list must be in range; 'tabulateList'
  -- only ever supplies in-range indices.
  fromInts :: [Int] -> HList (ToFinites shape)

instance HasIndex '[] where
  dims _ = []
  toInts HNil = []
  fromInts _ = HNil

instance (KnownNat (ToNat x), HasIndex xs) => HasIndex (x ': xs) where
  dims _ = natValI @(ToNat x) : dims (Proxy @xs)
  toInts (i :. is) = fromIntegral (getFinite i) : toInts @xs is
  fromInts (i : is) = case packFinite (fromIntegral i) of
    Just f -> f :. fromInts @xs is
    Nothing ->
      error $
        "Torch.Typed.Representable.fromInts: index "
          <> show i
          <> " is out of range for a dimension of size "
          <> show (natValI @(ToNat x))
  fromInts [] =
    error "Torch.Typed.Representable.fromInts: index list is shorter than the shape"

-- | The runtime dimensions of a named tensor's shape.
dimsOf ::
  forall shape dtype device.
  (HasIndex shape) =>
  Proxy (NamedTensor device dtype shape) ->
  [Int]
dimsOf _ = dims (Proxy @shape)

-- | Containers that are isomorphic to a function from an index.
class Representable t where
  -- | The index ("logarithm") of the container.
  type Log t :: Type

  -- | The element type held at each index.
  type Elem t :: Type

  index :: t -> Log t -> Elem t
  tabulate :: (Log t -> Elem t) -> t

instance
  ( HasIndex shape,
    TensorOptions (ToNats shape) dtype device,
    D.TensorLike (ComputeHaskellType dtype)
  ) =>
  Representable (NamedTensor device dtype shape)
  where
  type Log (NamedTensor device dtype shape) = HList (ToFinites shape)
  type Elem (NamedTensor device dtype shape) = ComputeHaskellType dtype

  index t = indexList t . toInts @shape
  tabulate f = tabulateList @shape @dtype @device (f . fromInts @shape)

-- | Read a single element, addressed by one index per dimension.
indexList ::
  forall shape dtype device.
  (D.TensorLike (ComputeHaskellType dtype)) =>
  NamedTensor device dtype shape ->
  [Int] ->
  ComputeHaskellType dtype
indexList t = D.asValue . foldl (\acc i -> D.select 0 i acc) (toDynamic t)

-- | Build a named tensor from a function on index lists.
tabulateList ::
  forall shape dtype device.
  ( HasIndex shape,
    TensorOptions (ToNats shape) dtype device,
    D.TensorLike (ComputeHaskellType dtype)
  ) =>
  ([Int] -> ComputeHaskellType dtype) ->
  NamedTensor device dtype shape
tabulateList f =
  fromUnnamed . UnsafeMkTensor . D.reshape ds $
    D.asTensor'
      (map f (allIndices ds))
      ( D.withDevice (optionsRuntimeDevice @(ToNats shape) @dtype @device)
          . D.withDType (optionsRuntimeDType @(ToNats shape) @dtype @device)
          $ D.defaultOpts
      )
  where
    ds = dims (Proxy @shape)

-- | Every index list for a shape, in row-major (C) order — matching the
-- element order that 'D.asTensor' plus 'D.reshape' produces.
allIndices :: [Int] -> [[Int]]
allIndices = foldr (\n acc -> [i : rest | i <- [0 .. n - 1], rest <- acc]) [[]]
