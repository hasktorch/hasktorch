{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | A graded (indexed) monad over tensor shapes.
--
-- Binding a tensor computation /concatenates/ shapes rather than preserving
-- them, so the shape cannot be a fixed parameter of a 'Monad' instance.  It is
-- instead a grade drawn from the monoid @('[], '(++)')@:
--
-- > greturn :: a           -> m '[]
-- > gbind   :: m s -> (a -> m t) -> m (s ++ t)
--
-- Because @'[]@ is a unit for @++@ and @++@ is associative, the graded monad
-- laws hold, and GHC discharges the required equalities definitionally — no
-- proof terms or coercions are needed.
--
-- This is deliberately /not/ a restricted monad.  A restricted monad would
-- keep the @Monad@ shape and attach constraints, which costs you @do@ notation
-- and every @Traversable@ combinator.  Here @do@ notation still works through
-- @QualifiedDo@ (GHC 9.0+):
--
-- > {-# LANGUAGE QualifiedDo #-}
-- > import qualified Torch.Typed.Graded as G
-- >
-- > rgbOf :: TensorMonad '( 'D.CPU, 0) 'D.Float '[Batch 2, RGB]
-- > rgbOf = G.do
-- >   x <- batch
-- >   channelwise x
--
-- The element type is fixed by the tensor's @dtype@, so unlike a
-- @Monad@ the grade parameter is the shape and the \"value\" type is constant.
-- That makes this a graded monad in the shape only, which is exactly the
-- structure tensor reshaping has.
module Torch.Typed.Graded
  ( -- * Shape concatenation
    --
    -- | Re-exported from "Torch.HList": the shape monoid's multiplication.
    type (++),

    -- * Graded monad
    GradedMonad (..),
    (>>=),
    (>>),
    return,

    -- * Tensors as a graded monad
    TensorMonad (..),
    fromNamed,
    toNamed,
  )
where

import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat)
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList (type (++))
import qualified Torch.Tensor as D
import Torch.Typed.Representable
import Torch.Typed.Tensor
import Prelude hiding (return, (>>), (>>=))

-- | A monad graded by a monoid of shapes.
--
-- The grading multiplication is 'Torch.HList.++', the type-level list
-- concatenation already used for 'HList' appending.
--
-- @m@ is indexed by a 'Shape' rather than by an element type, so 'gbind'
-- multiplies grades with '++' instead of leaving them fixed.
class GradedMonad (m :: Shape -> Type) where
  -- | The element type produced at each index.
  type Grade m :: Type

  -- | Whatever @m@ needs to know about a shape in order to build one.  For
  -- tensors this carries the runtime dimensions and tensor options.
  type Ok m (s :: Shape) :: Constraint

  -- | Unit of the grading: a scalar (rank-0) computation.
  greturn :: (Ok m '[]) => Grade m -> m '[]

  -- | Bind, concatenating the outer and inner shapes.  For each element of the
  -- outer tensor, @k@ produces an inner tensor; the results are spliced into a
  -- single tensor of the concatenated shape.
  gbind ::
    forall s t.
    (Ok m s, Ok m t, Ok m (s ++ t)) =>
    m s ->
    (Grade m -> m t) ->
    m (s ++ t)

-- | 'gbind' under the name @QualifiedDo@ desugars to.
(>>=) ::
  forall m s t.
  (GradedMonad m, Ok m s, Ok m t, Ok m (s ++ t)) =>
  m s ->
  (Grade m -> m t) ->
  m (s ++ t)
(>>=) = gbind

infixl 1 >>=

-- | Sequencing that ignores the bound element, for @QualifiedDo@ statements
-- without a binder.
(>>) ::
  forall m s t.
  (GradedMonad m, Ok m s, Ok m t, Ok m (s ++ t)) =>
  m s ->
  m t ->
  m (s ++ t)
a >> b = gbind a (const b)

infixl 1 >>

-- | 'greturn' under the name @QualifiedDo@ desugars to.
return :: forall m. (GradedMonad m, Ok m '[]) => Grade m -> m '[]
return = greturn

-- | A 'NamedTensor' with its device and dtype pinned, so that only the shape
-- remains as a type parameter and the result has kind @Shape -> Type@.
newtype TensorMonad (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (shape :: Shape)
  = TensorMonad (NamedTensor device dtype shape)

-- | Wrap a named tensor so it can be used with 'gbind'.
fromNamed :: NamedTensor device dtype shape -> TensorMonad device dtype shape
fromNamed = TensorMonad

-- | Unwrap back to a plain named tensor.
toNamed :: TensorMonad device dtype shape -> NamedTensor device dtype shape
toNamed (TensorMonad t) = t

instance
  (D.TensorLike (ComputeHaskellType dtype)) =>
  GradedMonad (TensorMonad device dtype)
  where
  type Grade (TensorMonad device dtype) = ComputeHaskellType dtype
  type
    Ok (TensorMonad device dtype) s =
      ( HasIndex s,
        TensorOptions (ToNats s) dtype device
      )

  greturn x = TensorMonad (tabulateList @'[] @dtype @device (const x))

  gbind ::
    forall s t.
    ( Ok (TensorMonad device dtype) s,
      Ok (TensorMonad device dtype) t,
      Ok (TensorMonad device dtype) (s ++ t)
    ) =>
    TensorMonad device dtype s ->
    (ComputeHaskellType dtype -> TensorMonad device dtype t) ->
    TensorMonad device dtype (s ++ t)
  gbind (TensorMonad outer) k =
    TensorMonad $
      tabulateList @(s ++ t) @dtype @device $ \ix ->
        -- an index into (s ++ t) splits into an outer and an inner index
        let (outerIx, innerIx) = splitAt rank ix
            x = indexList outer outerIx
         in indexList (toNamed (k x)) innerIx
    where
      rank = length (dims (Proxy @s))
