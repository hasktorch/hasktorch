{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.GraduallyTyped.NN.Functional.NonLinearActivation where

import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Shape (By (..), Dim, DimType, SelectDim (..), Shape (..), WithSelectDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen

type family SoftmaxF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  SoftmaxF 'UncheckedSelectDim _ = 'UncheckedShape
  SoftmaxF _ 'UncheckedShape = 'UncheckedShape
  SoftmaxF ( 'SelectDim by) ( 'Shape dims) = 'Shape dims

-- | Applies a softmax function that is defined as:
--
-- \[
-- \mathrm{Softmax}(\mathrm{input}_{i}) = \exp
-- \]
--
-- Softmax is applied to all slices along 'selectDim',
-- and will re-scale them so that the elements lie in the range \([0, 1]\) and sum to \(1\).
threshold ::
  forall selectDim requiresGradient layout device dataType shape.
  ( WithSelectDimC
      selectDim
      ( Tensor requiresGradient layout device dataType shape ->
        Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
      )
  ) =>
  WithSelectDimF
    selectDim
    ( Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
    )
threshold = withSelectDim @selectDim
  @( Tensor requiresGradient layout device dataType shape ->
     Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
   )
  $ \by tensor ->
    case by of
      ByName name -> unsafePerformIO $ cast2 ATen.softmax_tn tensor name
      ByIndex index -> unsafePerformIO $ cast2 ATen.softmax_tl tensor (fromInteger index :: Int)
