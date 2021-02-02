{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Comparison where

import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen

gt,
  lt,
  ge,
  le,
  eq,
  ne,
  (>.),
  (<.),
  (>=.),
  (<=.),
  (==.),
  (/=.) ::
    forall requiresGradient layout device dataType shape requiresGradient' layout' device' dataType' shape'.
    Tensor requiresGradient layout device dataType shape ->
    Tensor requiresGradient' layout' device' dataType' shape' ->
    Tensor
      'WithoutGradient
      (layout <+> layout')
      (device <+> device')
      (Seq (dataType <+> dataType') ( 'DataType 'Bool))
      (BroadcastShapesF shape shape')
a `gt` b = unsafePerformIO $ cast2 ATen.gt_tt a b
a `lt` b = unsafePerformIO $ cast2 ATen.lt_tt a b
a `ge` b = unsafePerformIO $ cast2 ATen.ge_tt a b
a `le` b = unsafePerformIO $ cast2 ATen.le_tt a b
a `eq` b = unsafePerformIO $ cast2 ATen.eq_tt a b
a `ne` b = unsafePerformIO $ cast2 ATen.ne_tt a b
(>.) = gt
(<.) = lt
(>=.) = ge
(<=.) = le
(==.) = eq
(/=.) = ne
