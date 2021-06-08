{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.Sparse where

import GHC.Natural (Natural)
import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Layout (KnownLayout, LayoutType (..))
import Torch.GraduallyTyped.Prelude (Reverse, Seq)
import Torch.GraduallyTyped.Shape (Dim (..), Name, Shape (..), Size)
import Torch.GraduallyTyped.Tensor.Type (Tensor, layout)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.Internal.Cast (cast5)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

type EmbedDimsErrorMessage (embedDims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply the embedding."
    % "The embedding weight tensor must have exactly two dimensions,"
    % "but the following dimensions were found:"
    % ""
    % "    " <> embedDims <> "."
    % ""

type family EmbeddingF (weightShape :: Shape [Dim (Name Symbol) (Size Nat)]) (inputShape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  EmbeddingF 'UncheckedShape _ = 'UncheckedShape
  EmbeddingF _ 'UncheckedShape = 'UncheckedShape
  EmbeddingF ( 'Shape '[_embedNumDim, embedDim]) ( 'Shape inputDims) = 'Shape (Reverse (embedDim ': Reverse inputDims))
  EmbeddingF ( 'Shape embedDims) _ = TypeError (EmbedDimsErrorMessage embedDims)

embedding ::
  forall requiresGradient layout device dataType shape requiresGradient' layout' device' dataType' shape'.
  (KnownLayout layout) =>
  -- | padding index
  Maybe Natural ->
  -- | whether or not to scale gradients by the inverse of frequency of the words in the mini-batch
  Bool ->
  -- | weight
  Tensor requiresGradient layout device dataType shape ->
  -- | input
  Tensor requiresGradient' layout' device' dataType' shape' ->
  -- | output
  Tensor
    (requiresGradient <|> requiresGradient')
    (layout <+> layout')
    (device <+> device')
    (Seq (dataType' <+> 'DataType 'Int64) dataType)
    (EmbeddingF shape shape')
embedding paddingIdx scaleGradByFreq weight input =
  let isSparse = layout weight == Sparse
      paddingIdx' :: Int = maybe (-1) fromIntegral paddingIdx
   in unsafePerformIO $ cast5 ATen.embedding_ttlbb weight input paddingIdx' scaleGradByFreq isSparse
