{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.Linear where

import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (UnifyDataTypeF)
import Torch.GraduallyTyped.Device (UnifyDeviceF)
import Torch.GraduallyTyped.Layout (UnifyLayoutF)
import Torch.GraduallyTyped.Prelude (Reverse, Seq)
import Torch.GraduallyTyped.Shape (Dim (..), Name, Shape (..), Size, UnifyDimF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast3)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

-- | Compute the output shape of a linear transformation.
--
-- >>> type InputDim = 'Dim ('Name "input") ('Size 5)
-- >>> type OutputDim = 'Dim ('Name "output") ('Size 10)
-- >>> type BatchDim = 'Dim ('Name "batch") ('Size 20)
-- >>> type WeightShape = 'Shape '[OutputDim, InputDim]
-- >>> type BiasShape = 'Shape '[OutputDim]
-- >>> type InputShape = 'Shape '[BatchDim, InputDim]
-- >>> :kind! LinearF WeightShape BiasShape InputShape
-- LinearF WeightShape BiasShape InputShape :: Shape [Dim (Name Symbol) (Size Nat)]
-- = 'Shape
--     '[ 'Dim ('Name "batch") ('Size 20),
--        'Dim ('Name "output") ('Size 10)]
type family LinearF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (biasShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  LinearF ( 'Shape '[]) _ _ = TypeError (LinearWeightDimsErrorMessage '[])
  LinearF ( 'Shape '[weightDim]) _ _ = TypeError (LinearWeightDimsErrorMessage '[weightDim])
  LinearF ( 'Shape (weightDim ': weightDim' ': weightDim'' ': weightDims)) _ _ = TypeError (LinearWeightDimsErrorMessage (weightDim ': weightDim' ': weightDim'' ': weightDims))
  LinearF _ ( 'Shape '[]) _ = TypeError (LinearBiasDimsErrorMessage '[])
  LinearF _ ( 'Shape (biasDim ': biasDim' ': biasDims)) _ = TypeError (LinearBiasDimsErrorMessage (biasDim ': biasDim' ': biasDims))
  LinearF _ _ ( 'Shape '[]) = TypeError LinearInputDimsErrorMessage
  LinearF ( 'Shape weightDims) ( 'Shape biasDims) ( 'Shape inputDims) = 'Shape (Reverse (LinearDimsF weightDims biasDims (Reverse inputDims)))
  LinearF 'UncheckedShape _ _ = 'UncheckedShape
  LinearF _ 'UncheckedShape _ = 'UncheckedShape
  LinearF _ _ 'UncheckedShape = 'UncheckedShape

type family LinearDimsF (weightDims :: [Dim (Name Symbol) (Size Nat)]) (biasDims :: [Dim (Name Symbol) (Size Nat)]) (reversedInputDims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  LinearDimsF '[outputDim, inputDim] '[outputDim'] (inputDim' ': reversedInputDims) = Seq (UnifyDimF inputDim inputDim') (UnifyDimF outputDim outputDim' ': reversedInputDims)

type LinearInputDimsErrorMessage =
  "Cannot apply the linear transformation."
    % "The input tensor does not have the minimum required number of dimensions."
    % "At least one dimension is needed, but none were found."

type LinearBiasDimsErrorMessage (biasDims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply the linear transformation."
    % "The bias tensor must have exactly one dimension,"
    % "but the following dimensions were found:"
    % ""
    % "    " <> biasDims <> "."
    % ""

type LinearWeightDimsErrorMessage (weightDims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply the linear transformation."
    % "The weight tensor must have exactly two dimensions,"
    % "but the following dimensions were found:"
    % ""
    % "    " <> weightDims <> "."
    % ""

-- | Applies a linear transformation to the incoming data:
-- \[
-- \mathrm{output} = \mathrm{input} \mathrm{weight}^{\intercal} + \mathrm{bias}.
-- \]
--
-- Supported shapes:
--
--     * 'input': \((N, \ldots, \mathrm{inputFeatures})\), where \(N\) is the batch size,
--     \(\ldots\) means any number of additional dimensions and
--     \(\mathrm{inputFeatures}\) are the input features.
--
--     * 'weight': \((\mathrm{outputFeatures}, \mathrm{inputFeatures})\)
--
--     * 'bias': \((\mathrm{outputFeatures})\)
--
--     * 'output': \((N, \ldots, \mathrm{outputFeatures})\)
--
-- Examples:
--
-- >>> type InputDim = 'Dim ('Name "input") ('Size 5)
-- >>> type OutputDim = 'Dim ('Name "output") ('Size 10)
-- >>> type BatchDim = 'Dim ('Name "batch") ('Size 20)
-- >>> type WeightShape = 'Shape '[OutputDim, InputDim]
-- >>> type BiasShape = 'Shape '[OutputDim]
-- >>> type InputShape = 'Shape '[BatchDim, InputDim]
-- >>> g <- generator @('Device 'CPU) 0
-- >>> (weight, g') = randn @'Independent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @WeightShape g
-- >>> (bias, g'') = randn @'Independent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @BiasShape g'
-- >>> (input, _) = randn @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @InputShape g''
-- >>> result = linear weight bias input
-- >>> :type result
-- result
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 20),
--              'Dim ('Name "output") ('Size 10)])
linear ::
  forall requiresGradient requiresGradient' requiresGradient'' layout layout' layout'' device device' device'' dataType dataType' dataType'' shape shape' shape''.
  -- | weight
  Tensor requiresGradient layout device dataType shape ->
  -- | bias
  Tensor requiresGradient' layout' device' dataType' shape' ->
  -- | input
  Tensor requiresGradient'' layout'' device'' dataType'' shape'' ->
  -- | output
  Tensor
    requiresGradient''
    (UnifyLayoutF (UnifyLayoutF layout'' layout) layout')
    (UnifyDeviceF (UnifyDeviceF device'' device) device')
    (UnifyDataTypeF (UnifyDataTypeF dataType'' dataType) dataType')
    (LinearF shape shape' shape'')
linear weight bias input = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias