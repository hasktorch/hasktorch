{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.Normalization where

import GHC.TypeLits (Nat, Symbol, TypeError, type (+), type (-))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Prelude (Length, Reverse)
import Torch.GraduallyTyped.Scalar ()
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SelectDims (..), Shape (..), Size (..), dimSize)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (getDims), Tensor (..))
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
  LayerNormWithBiasF ('Shape weightDims) ('Shape biasDims) ('Shape inputDims) = 'Shape (Reverse (LayerNormImplF (Reverse (weightDims <+> biasDims)) (Reverse inputDims)))

layerNormWithBias ::
  forall gradient gradient' gradient'' layout layout' layout'' device device' device'' dataType dataType' dataType'' shape shape' shape''.
  SGetShape shape =>
  -- | weight
  Tensor gradient layout device dataType shape ->
  -- | bias
  Tensor gradient' layout' device' dataType' shape' ->
  -- | eps
  Double ->
  -- | input
  Tensor gradient'' layout'' device'' dataType'' shape'' ->
  -- | output
  Tensor
    (gradient' <|> gradient' <|> gradient'')
    (layout <+> layout' <+> layout'')
    (device <+> device' <+> device'')
    (dataType <+> dataType' <+> dataType'')
    (LayerNormWithBiasF shape shape' shape'')
layerNormWithBias weight bias eps input = unsafePerformIO $ do
  let weightDims = getDims weight
  cast5 ATen.layer_norm_tlttd input (dimSize <$> weightDims) weight bias eps

type family LayerNormWithoutBiasF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LayerNormWithoutBiasF 'UncheckedShape _ = 'UncheckedShape
  LayerNormWithoutBiasF _ 'UncheckedShape = 'UncheckedShape
  LayerNormWithoutBiasF ('Shape weightDims) ('Shape inputDims) = 'Shape (Reverse (LayerNormImplF (Reverse weightDims) (Reverse inputDims)))

type family LayerNormWithoutBiasSelectDimsF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: SelectDims [By Symbol Nat] where
  LayerNormWithoutBiasSelectDimsF 'UncheckedShape _ = 'UncheckedSelectDims
  LayerNormWithoutBiasSelectDimsF _ 'UncheckedShape = 'UncheckedSelectDims
  LayerNormWithoutBiasSelectDimsF ('Shape weightDims) ('Shape inputDims) = 'SelectDims (LayerNormWithoutBiasBysF weightDims inputDims (Length inputDims) 1)

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
  forall gradient layout device dataType shape gradient' layout' device' dataType' shape'.
  (SGetShape shape, SGetShape shape') =>
  -- | weight
  Tensor gradient layout device dataType shape ->
  -- | eps
  Double ->
  -- | input
  Tensor gradient' layout' device' dataType' shape' ->
  -- | output
  Tensor
    (gradient <|> gradient')
    (layout <+> layout')
    (device <+> device')
    (dataType <+> dataType')
    (LayerNormWithoutBiasF shape shape')
layerNormWithoutBias weight eps input = unsafePerformIO $ do
  let weightDims = getDims weight
      inputDims = getDims input
  let indexes :: [Int] = fromIntegral . (length inputDims -) <$> [1, 2 .. length weightDims]
  cast6 (go (null indexes)) input weight indexes eps (2 :: Double) True
  where
    go nullIndexes input' weight' indexes eps' exponent' keepDim = do
      squaredInput <- ATen.pow_ts input' exponent'
      variance <-
        if nullIndexes
          then pure squaredInput
          else
            ATen.mean_tlb
              squaredInput
              indexes
              keepDim
      ATen.add_ts variance eps'
        >>= ATen.rsqrt_t
        >>= ATen.mul_tt input'
        >>= ATen.mul_tt weight'
