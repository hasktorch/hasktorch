{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.NonLinearActivation where

import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Shape (By (..), Dim, GetDimImplF, Name, SelectDim (..), Shape (..), Size, WithSelectDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

type SoftMaxErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply softmax on the dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family SoftmaxCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe (Dim (Name Symbol) (Size Nat))) :: [Dim (Name Symbol) (Size Nat)] where
  SoftmaxCheckF by dims 'Nothing = TypeError (SoftMaxErrorMessage by dims)
  SoftmaxCheckF _ dims ( 'Just _) = dims

type family SoftmaxF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  SoftmaxF 'UncheckedSelectDim _ = 'UncheckedShape
  SoftmaxF _ 'UncheckedShape = 'UncheckedShape
  SoftmaxF ( 'SelectDim by) ( 'Shape dims) = 'Shape (SoftmaxCheckF by dims (GetDimImplF by dims))

-- | Applies the softmax function that is defined as:
--
-- \[
-- \mathrm{Softmax}(\mathrm{input}_{i}) = \frac{\exp\left(\mathrm{input}_{i}\right)}{\sum_j \exp\left(\mathrm{input}_{j}\right)}
-- \]
--
-- Softmax is applied to all slices along 'selectDim',
-- and will re-scale them so that the elements lie in the range \([0, 1]\) and sum to \(1\):
--
-- >>> g <- mkGenerator @('Device 'CPU) 0
-- >>> (input, _) = randn @'Dependent @('Layout Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) g
-- >>> result = softmax @('SelectDim ('ByName "feature")) input
-- >>> :type result
-- result
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
softmax, logSoftmax ::
  forall selectDim requiresGradient layout device dataType shape.
  WithSelectDimC
    selectDim
    ( Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
    ) =>
  WithSelectDimF
    selectDim
    ( Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
    )
softmax = withSelectDim @selectDim
  @( Tensor requiresGradient layout device dataType shape ->
     Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
   )
  $ \by tensor ->
    case by of
      ByName name -> unsafePerformIO $ cast2 ATen.softmax_tn tensor name
      ByIndex index -> unsafePerformIO $ cast2 ATen.softmax_tl tensor (fromInteger index :: Int)
logSoftmax = withSelectDim @selectDim
  @( Tensor requiresGradient layout device dataType shape ->
     Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
   )
  $ \by tensor ->
    case by of
      ByName name -> unsafePerformIO $ cast2 ATen.log_softmax_tn tensor name
      ByIndex index -> unsafePerformIO $ cast2 ATen.log_softmax_tl tensor (fromInteger index :: Int)
