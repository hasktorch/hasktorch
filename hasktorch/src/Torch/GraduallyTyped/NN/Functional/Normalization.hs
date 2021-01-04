{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.Normalization where

import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Reverse)
import Torch.GraduallyTyped.Shape (Dim, KnownShape, Name, Shape (..), Size, dimSize)
import Torch.GraduallyTyped.Tensor.Type (Tensor, shape)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast5)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%))

type family LayerNormImplF (reverseNormalizedDims :: [Dim (Name Symbol) (Size Nat)]) (reverseInputDims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  LayerNormImplF '[] reverseInputDims = reverseInputDims
  LayerNormImplF (normalizedDim ': reverseNormalizedDims) (inputDim ': reverseInputDims) = normalizedDim <+> inputDim ': LayerNormImplF reverseNormalizedDims reverseInputDims
  LayerNormImplF _ '[] = TypeError LayerNormShapeErrorMessage

type LayerNormShapeErrorMessage =
  "Cannot apply layer norm. "
    % "Normalized shape exceeds input shape."

type family LayerNormF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (biasShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LayerNormF 'UncheckedShape _ _ = 'UncheckedShape
  LayerNormF _ 'UncheckedShape _ = 'UncheckedShape
  LayerNormF _ _ 'UncheckedShape = 'UncheckedShape
  LayerNormF ( 'Shape weightDims) ( 'Shape biasDims) ( 'Shape inputDims) = 'Shape (Reverse (LayerNormImplF (Reverse (weightDims <+> biasDims)) (Reverse inputDims)))

layerNorm ::
  forall requiresGradient requiresGradient' requiresGradient'' layout layout' layout'' device device' device'' dataType dataType' dataType'' shape shape' shape''.
  (KnownShape shape) =>
  -- | weight
  Tensor requiresGradient layout device dataType shape ->
  -- | bias
  Tensor requiresGradient' layout' device' dataType' shape' ->
  -- | eps
  Double ->
  -- | input
  Tensor requiresGradient'' layout'' device'' dataType'' shape'' ->
  -- | output
  Tensor
    requiresGradient''
    (layout <+> (layout' <+> layout''))
    (device <+> (device' <+> device''))
    (dataType <+> (dataType' <+> dataType''))
    (LayerNormF shape shape' shape'')
layerNorm weight bias eps input =
  let dims = shape weight
   in unsafePerformIO $
        cast5 ATen.layer_norm_tlttd input (dimSize <$> dims) weight bias eps