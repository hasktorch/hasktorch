{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Staged element-wise operations: write per-element code once, run it
-- vectorized.
--
-- The trick is Coyoneda-shaped, with the function space restricted to what can
-- be reinterpreted: element functions are written polymorphically,
--
-- > f :: (Floating a, Cond a) => a -> a
-- > f x = whereE (gtE x 0) (sin x * 10) 0
--
-- and the /same/ code runs under two instantiations:
--
-- * @a = Float@ — reference semantics, one call per element (what
--   'Torch.Typed.Representable.tabulate' and 'Torch.Typed.Graded.gbind' do);
-- * @a = Tensor@ — each operation is a whole-tensor ATen call, so the function
--   body executes /once/ regardless of tensor size, autograd sees every step,
--   and no per-element FFI happens.
--
-- Because the argument type is universally quantified, the only operations
-- available inside @f@ are the class methods — parametricity guarantees the
-- function is a pointwise arithmetic expression, so reinterpreting it at
-- @Tensor@ is sound.
--
-- An opaque, already-monomorphic @Float -> Float@ cannot be vectorized this
-- way (GHC has no runtime code inspection); the polymorphic type is what keeps
-- the code inspectable.
module Torch.Typed.Staged
  ( -- * Value-dependent control flow
    --
    -- | Re-exported from "Torch.Elementwise", their untyped home: the same
    -- classes drive element-wise code over untyped tensors.
    Cond (..),

    -- * Vectorized element-wise operations
    emap,
    ezipWith,

    -- * Structured elements
    --
    -- | 'emapS' and 'ezipWithS' generalize 'emap'\/'ezipWith' from scalar
    -- elements to /structured/ elements: the trailing dimensions of a tensor
    -- are read as the inside of a compound element.  A batch of triangles,
    -- each described by three vertices with RGB values, is
    -- @'[Batch n, Vector 3, RGB]@ — or equally a tensor of shape
    -- @'[Batch n]@ whose elements are @Vector 3 (RGB a)@.  The 'Nested'
    -- family maps between the two readings.
    Nested,
    ExplodeShape (..),
    emapS,
    ezipWithS,

    -- * Staged graded bind
    gbindV,
  )
where

import Data.Default.Class (Default (..))
import Data.Kind (Type)
import Data.Maybe (fromJust)
import Data.Proxy (Proxy (..))
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import GHC.TypeLits (KnownNat)
import Torch.Elementwise (Cond (..))
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import Torch.HList (HList, type (++))
import Torch.Lens (HasTypes, flattenValues, replaceValues, types)
import qualified Torch.Tensor as D
import Torch.Typed.Graded (TensorMonad, fromNamed, toNamed)
import Torch.Typed.Representable
import Torch.Typed.Tensor

-- | Element-wise map, executed as whole-tensor operations.
--
-- Semantically @emap f t = tabulate (f . index t)@, but @f@ runs once at
-- @a = Tensor@ instead of once per element.
emap ::
  forall shape dtype device.
  (forall a. (Floating a, Cond a) => a -> a) ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape
emap f = fromUnnamed . UnsafeMkTensor . f . toDynamic

-- | Element-wise combination of two same-shaped tensors, executed as
-- whole-tensor operations.
ezipWith ::
  forall shape dtype device.
  (forall a. (Floating a, Cond a) => a -> a -> a) ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape
ezipWith f x y = fromUnnamed . UnsafeMkTensor $ f (toDynamic x) (toDynamic y)

-- | The element type a trailing shape describes: @Nested '[Vector 3, RGB] a@
-- is @Vector 3 (RGB a)@.  Because the family reduces structurally, formulas
-- may be written against ordinary type synonyms like
-- @type Triangle a = Vector 3 (RGB a)@.
type family Nested (shape :: Shape) (a :: Type) :: Type where
  Nested '[] a = a
  Nested (f ': fs) a = f (Nested fs a)

-- | Shapes whose dimensions can be unrolled into (and re-rolled from) actual
-- Haskell structure, one tensor component per position.  The tensors handled
-- here always keep one leading batch dimension; 'explode' peels the trailing
-- dimensions off into structure, so each component has the batch dimension
-- only.
class ExplodeShape (sh :: Shape) where
  explode :: D.Tensor -> Nested sh D.Tensor
  implode :: Nested sh D.Tensor -> D.Tensor

instance ExplodeShape '[] where
  explode = id
  implode = id

instance (KnownNat n, ExplodeShape fs) => ExplodeShape (Vector n ': fs) where
  explode t = fromJust . V.fromList $ map (explode @fs) (I.unbind t 1)
  implode v = F.stack (F.Dim 1) (map (implode @fs) (V.toList v))

instance
  {-# OVERLAPS #-}
  ( Default (g (Nested fs D.Tensor)),
    HasTypes (g (Nested fs D.Tensor)) (Nested fs D.Tensor),
    ExplodeShape fs
  ) =>
  ExplodeShape (g ': fs)
  where
  explode t = replaceValues (types @(Nested fs D.Tensor)) def (map (explode @fs) (I.unbind t 1))
  implode s = F.stack (F.Dim 1) (map (implode @fs) (flattenValues (types @(Nested fs D.Tensor)) s))

-- | 'emap' with structured elements.  The trailing dimensions @sh@ of the
-- input are unrolled into the Haskell structure @'Nested' sh a@, the
-- polymorphic function runs /once/ at @a = Tensor@ — every scalar operation
-- in it becomes one whole-batch kernel — and the resulting structure's
-- dimensions @sh'@ are rolled back into the tensor.  The structure may change
-- shape: averaging the channels of each vertex is
--
-- > emapS (fmap (\(RGB r g b) -> (r + g + b) / 3))
-- >   :: NamedTensor device dtype '[Batch n, Vector 3, RGB]
-- >   -> NamedTensor device dtype '[Batch n, Vector 3]
--
-- and structural operations (folds over vertices, permutations, pattern
-- matching on fields) are ordinary Haskell on the structure, still vectorized
-- over the batch.
--
-- Cost: the structure is unrolled positionally — @numel sh@ @select@s in, one
-- kernel per scalar operation in the formula, @numel sh'@ tensors stacked
-- out.  This is the fusion of 'emap' extended to compound elements, and like
-- 'gbindV' it suits /small, static/ element structures (vertices, channels,
-- gates); it is not for structures with thousands of positions.  Where
-- 'Torch.Typed.Representable.vmap' calls an opaque slice function once per
-- batch position, 'emapS' runs a transparent one once in total.
--
-- The result shape @sh'@ is generally not inferable from the formula (the
-- 'Nested' family is not injective), so annotate the result or apply the
-- shapes with type applications: @emapS \@sh \@sh'@.
emapS ::
  forall sh sh' b dtype device.
  (ExplodeShape sh, ExplodeShape sh') =>
  (forall a. (Floating a, Cond a) => Nested sh a -> Nested sh' a) ->
  NamedTensor device dtype (b ': sh) ->
  NamedTensor device dtype (b ': sh')
emapS f =
  fromUnnamed . UnsafeMkTensor . implode @sh' . f @D.Tensor . explode @sh . toDynamic

-- | 'ezipWith' with structured elements: the two-argument 'emapS'.  The two
-- inputs share the batch dimension but may have different element structures.
ezipWithS ::
  forall shA shB sh' b dtype device.
  (ExplodeShape shA, ExplodeShape shB, ExplodeShape sh') =>
  (forall a. (Floating a, Cond a) => Nested shA a -> Nested shB a -> Nested sh' a) ->
  NamedTensor device dtype (b ': shA) ->
  NamedTensor device dtype (b ': shB) ->
  NamedTensor device dtype (b ': sh')
ezipWithS f x y =
  fromUnnamed . UnsafeMkTensor . implode @sh' $
    f @D.Tensor (explode @shA (toDynamic x)) (explode @shB (toDynamic y))

-- | Staged version of 'Torch.Typed.Graded.gbind'.  Definitionally,
--
-- > gbindV m k = gbind m (\x -> fromNamed (tabulate (k x)))
--
-- (this equation is checked in the test suite, with
-- 'Torch.Typed.Graded.gbind' as the oracle), but where 'Torch.Typed.Graded.gbind'
-- evaluates @k@ per element, 'gbindV' evaluates it once per /inner index/ on
-- the whole outer tensor: the cost is @numel t@ chains of ATen calls instead
-- of @numel s * numel t@ per-element FFI round trips, and gradients flow
-- through every step.
--
-- The inner indices are enumerated concretely, so this suits small inner
-- dimensions (channels, coordinates, classes).  A large inner dimension would
-- need a symbolic index (@arange@-based) instead.
gbindV ::
  forall t s device dtype.
  (HasIndex s, HasIndex t) =>
  TensorMonad device dtype s ->
  (forall a. (Floating a, Cond a) => a -> HList (ToFinites t) -> a) ->
  TensorMonad device dtype (s ++ t)
gbindV m k =
  fromNamed . fromUnnamed . UnsafeMkTensor $
    D.reshape (outerDims ++ innerDims) stacked
  where
    x = toDynamic (toNamed m)
    outerDims = dims (Proxy @s)
    innerDims = dims (Proxy @t)
    -- one whole-tensor evaluation of k per inner index, row-major, matching
    -- the element order of the reference gbind
    stacked =
      F.stack
        (F.Dim (length outerDims))
        [k x (fromInts @t ix) | ix <- allIndices innerDims]
