{-# LANGUAGE RankNTypes #-}

-- | Element-wise code over untyped tensors, written once and interpreted
-- twice.
--
-- The untyped 'Torch.Tensor.Tensor' has 'Num', 'Fractional' and 'Floating'
-- instances, so a function written polymorphically,
--
-- > f :: (Floating a, Cond a) => a -> a
-- > f x = whereE (gtE x 0) (sin x * 10) 0
--
-- runs under two instantiations: at @a = Float@ one element at a time (the
-- reference semantics), and at @a = 'Torch.Tensor.Tensor'@ as whole-tensor
-- ATen operations — the body executes once regardless of tensor size, no
-- per-element FFI happens, and autograd sees every step.
--
-- 'Cond' supplies the part 'Floating' cannot: value-dependent control flow.
-- @if@ on the element is not available at @a = Tensor@ (there is no 'Bool' to
-- inspect), so branching goes through 'whereE', which compiles to
-- @torch.where@.
--
-- The typed layer ("Torch.Typed.Staged") builds on exactly these classes and
-- adds shape tracking; this module is their untyped home.
module Torch.Elementwise
  ( -- * Value-dependent control flow
    Cond (..),

    -- * Vectorized element-wise application
    emap,
    ezipWith,
  )
where

import qualified Torch.DType as DT
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import Torch.Tensor (Tensor)
import qualified Torch.Tensor as T

-- | Branching and comparisons for staged element code.  Comparisons return
-- @0@\/@1@ masks in the same type @a@ so they compose with arithmetic.
class Cond a where
  -- | @whereE c t e@ is elementwise @if c /= 0 then t else e@.
  whereE :: a -> a -> a -> a

  ltE :: a -> a -> a
  leE :: a -> a -> a
  gtE :: a -> a -> a
  geE :: a -> a -> a
  eqE :: a -> a -> a
  neE :: a -> a -> a

  -- | Elementwise maximum.  The default goes through 'whereE'; instances can
  -- (and the 'Tensor' instance does) override it with a native operation.
  maxE :: a -> a -> a
  maxE a b = whereE (gtE a b) a b

  -- | Elementwise minimum, see 'maxE'.
  minE :: a -> a -> a
  minE a b = whereE (ltE a b) a b

instance Cond Float where
  whereE c t e = if c /= 0 then t else e
  ltE a b = if a < b then 1 else 0
  leE a b = if a <= b then 1 else 0
  gtE a b = if a > b then 1 else 0
  geE a b = if a >= b then 1 else 0
  eqE a b = if a == b then 1 else 0
  neE a b = if a /= b then 1 else 0

instance Cond Double where
  whereE c t e = if c /= 0 then t else e
  ltE a b = if a < b then 1 else 0
  leE a b = if a <= b then 1 else 0
  gtE a b = if a > b then 1 else 0
  geE a b = if a >= b then 1 else 0
  eqE a b = if a == b then 1 else 0
  neE a b = if a /= b then 1 else 0

instance Cond Tensor where
  whereE c t e = I.where' (F.toDType DT.Bool c) t e
  ltE a b = F.toDType (T.dtype a) (F.lt a b)
  leE a b = F.toDType (T.dtype a) (F.le a b)
  gtE a b = F.toDType (T.dtype a) (F.gt a b)
  geE a b = F.toDType (T.dtype a) (F.ge a b)
  eqE a b = F.toDType (T.dtype a) (F.eq a b)
  neE a b = F.toDType (T.dtype a) (F.ne a b)
  maxE = I.maximum
  minE = I.minimum

-- | Apply an element function to a whole tensor.  Operationally this is just
-- application — the function /is/ the vectorized code — but the rank-2 type
-- earns its keep: the only operations available inside @f@ are class methods,
-- so parametricity guarantees @f@ is a pointwise expression, and the same
-- @f@ can be run at @a = Float@ as its own per-element specification.
emap :: (forall a. (Floating a, Cond a) => a -> a) -> Tensor -> Tensor
emap f = f

-- | Element-wise combination of two tensors (with broadcasting, as the
-- underlying ATen operations provide it).
ezipWith :: (forall a. (Floating a, Cond a) => a -> a -> a) -> Tensor -> Tensor -> Tensor
ezipWith f = f
