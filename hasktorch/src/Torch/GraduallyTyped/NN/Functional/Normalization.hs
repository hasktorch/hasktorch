{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.Normalization where

import GHC.TypeLits (Nat, Symbol, TypeError, type (+), type (-))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Length, Reverse)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim, KnownShape, Name, SelectDims (..), Shape (..), Size, WithSelectDimsC (..), dimSize)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar, mul, powScalar, rsqrt)
import Torch.GraduallyTyped.Tensor.MathOperations.Reduction (MeanF, mean)
import Torch.GraduallyTyped.Tensor.Type (Tensor, shape)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.Internal.Cast (cast5, cast6)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

type family LayerNormImplF (reverseNormalizedDims :: [Dim (Name Symbol) (Size Nat)]) (reverseInputDims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  LayerNormImplF '[] reverseInputDims = reverseInputDims
  LayerNormImplF (normalizedDim ': reverseNormalizedDims) (inputDim ': reverseInputDims) = normalizedDim <+> inputDim ': LayerNormImplF reverseNormalizedDims reverseInputDims
  LayerNormImplF _ '[] = TypeError LayerNormShapeErrorMessage

type LayerNormShapeErrorMessage =
  "Cannot apply the layer norm. "
    % "The normalized shape exceeds the input shape."

type family LayerNormWithBiasF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (biasShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LayerNormWithBiasF 'UncheckedShape _ _ = 'UncheckedShape
  LayerNormWithBiasF _ 'UncheckedShape _ = 'UncheckedShape
  LayerNormWithBiasF _ _ 'UncheckedShape = 'UncheckedShape
  LayerNormWithBiasF ( 'Shape weightDims) ( 'Shape biasDims) ( 'Shape inputDims) = 'Shape (Reverse (LayerNormImplF (Reverse (weightDims <+> biasDims)) (Reverse inputDims)))

layerNormWithBias ::
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
    (requiresGradient' <|> (requiresGradient' <|> requiresGradient''))
    (layout <+> (layout' <+> layout''))
    (device <+> (device' <+> device''))
    (dataType <+> (dataType' <+> dataType''))
    (LayerNormWithBiasF shape shape' shape'')
layerNormWithBias weight bias eps input =
  let dims = shape weight
   in unsafePerformIO $
        cast5 ATen.layer_norm_tlttd input (dimSize <$> dims) weight bias eps

type family LayerNormWithoutBiasF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LayerNormWithoutBiasF 'UncheckedShape _ = 'UncheckedShape
  LayerNormWithoutBiasF _ 'UncheckedShape = 'UncheckedShape
  LayerNormWithoutBiasF ( 'Shape weightDims) ( 'Shape inputDims) = 'Shape (Reverse (LayerNormImplF (Reverse weightDims) (Reverse inputDims)))

type family LayerNormWithoutBiasF' (selectDims :: SelectDims [By Symbol Nat]) (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LayerNormWithoutBiasF' selectDims weightShape inputShape =
    BroadcastShapesF
      weightShape
      ( BroadcastShapesF
          inputShape
          (MeanF selectDims inputShape)
      )

type family LayerNormWithoutBiasSelectDimsF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: SelectDims [By Symbol Nat] where
  LayerNormWithoutBiasSelectDimsF 'UncheckedShape _ = 'UncheckedSelectDims
  LayerNormWithoutBiasSelectDimsF _ 'UncheckedShape = 'UncheckedSelectDims
  LayerNormWithoutBiasSelectDimsF ( 'Shape weightDims) ( 'Shape inputDims) = 'SelectDims (LayerNormWithoutBiasBysF weightDims inputDims (Length inputDims) 1)

type family LayerNormWithoutBiasBysF (weightDims :: [Dim (Name Symbol) (Size Nat)]) (inputDims :: [Dim (Name Symbol) (Size Nat)]) (inputDimsLength :: Nat) (counter :: Nat) :: [By Symbol Nat] where
  LayerNormWithoutBiasBysF '[] _ _ _ = '[]
  LayerNormWithoutBiasBysF (_ ': weightDims) (_ ': inputDims) inputDimsLength counter = 'ByIndex (inputDimsLength - counter) ': LayerNormWithoutBiasBysF weightDims inputDims inputDimsLength (counter + 1)
  LayerNormWithoutBiasBysF _ '[] inputDimsLength counter =
    TypeError
      ( "Cannot apply the layer norm."
          % "The provided weight tensor has more dimensions than the input tensor,"
          % ""
          % "    '" <> counter <> "'"
          % ""
          % "and"
          % ""
          % "    '" <> inputDimsLength <> "',"
          % ""
          % "respectively."
      )

-- | T5-style layer norm
layerNormWithoutBias ::
  forall requiresGradient layout device dataType shape requiresGradient' layout' device' dataType' shape'.
  ( KnownShape shape,
    KnownShape shape'
  ) =>
  -- | weight
  Tensor requiresGradient layout device dataType shape ->
  -- | eps
  Double ->
  -- | input
  Tensor requiresGradient' layout' device' dataType' shape' ->
  -- | output
  Tensor
    (requiresGradient <|> requiresGradient')
    (layout <+> layout')
    (device <+> device')
    (dataType <+> dataType')
    (LayerNormWithoutBiasF shape shape')
layerNormWithoutBias weight eps input =
  let weightShape = shape weight
      inputShape = shape input
      indexes :: [Int] = fromIntegral . (length inputShape -) <$> [1, 2 .. length weightShape]
   in unsafePerformIO $
        cast6 (go (null indexes)) input weight indexes eps (2 :: Double) True
  where
    go nullIndexes input weight indexes eps exponent keepDim = do
      squaredInput <- ATen.pow_ts input exponent
      variance <-
        if nullIndexes
          then pure squaredInput
          else
            ATen.mean_tlb
              squaredInput
              indexes
              keepDim
      ATen.add_ts variance eps
        >>= ATen.rsqrt_t
        >>= ATen.mul_tt input
        >>= ATen.mul_tt weight

layerNormWithoutBias' ::
  forall requiresGradient layout device dataType shape requiresGradient' layout' device' dataType' shape' selectDims.
  ( KnownShape shape,
    KnownShape shape',
    selectDims ~ LayerNormWithoutBiasSelectDimsF shape shape',
    WithSelectDimsC
      selectDims
      ( Tensor requiresGradient' layout' device' dataType' shape' ->
        Tensor requiresGradient' layout' device' dataType' (MeanF selectDims shape')
      )
  ) =>
  -- | weight
  Tensor requiresGradient layout device dataType shape ->
  -- | eps
  Double ->
  -- | input
  Tensor requiresGradient' layout' device' dataType' shape' ->
  -- | output
  Tensor
    (requiresGradient <|> requiresGradient')
    (layout <+> layout')
    (device <+> device')
    (dataType <+> dataType')
    (LayerNormWithoutBiasF' selectDims shape shape')
layerNormWithoutBias' weight eps input =
  let weightShape = shape weight
      inputShape = shape input
      bys = ByIndex . fromIntegral . (length inputShape -) <$> [1, 2 .. length weightShape]
      variance =
        withoutSelectDims @selectDims @(Tensor requiresGradient' layout' device' dataType' shape' -> Tensor requiresGradient' layout' device' dataType' (MeanF selectDims shape'))
          ( mean @selectDims @requiresGradient' @layout' @device' @dataType' @shape'
          )
          bys
          (powScalar (2 :: Float) input)
   in weight `mul` (input `mul` rsqrt (variance `addScalar` eps))
