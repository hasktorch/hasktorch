{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -v2
                -fomit-interface-pragmas
                -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL1
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.T5 where

import qualified Data.Map as Map
import GHC.TypeLits (KnownNat, Nat)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer(..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (NumelDimF, NumelF, Shape (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), shape)
import Torch.Script (IValue (..))
import Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm(LayerNorm))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder(TransformerEncoder))
import Torch.GraduallyTyped.NN (HasInitialize(initialize))

-- | num_layers = 6
type T5SmallNumLayers = 6

-- | n_heads = 8
type T5SmallHeadDim = 'Dim ( 'Name "head") ( 'Size 8)

-- | d_kv = 64
type T5SmallHeadEmbedDim = 'Dim ( 'Name "headEmbed") ( 'Size 64)

-- | inner_dim =  = n_heads * d_kv = 512
type T5SmallEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_model = 512
type T5SmallInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_model = 512
type T5SmallDecoderInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_ff = 2048
type T5SmallFFNDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- | https://huggingface.co/t5-small/blob/main/config.json
data T5Small device dataType where
  T5Small ::
    forall device dataType.
    SequenceToSequenceTransformer
      T5SmallNumLayers
      T5SmallNumLayers
      device
      dataType
      T5SmallHeadDim
      T5SmallHeadEmbedDim
      T5SmallEmbedDim
      T5SmallInputEmbedDim
      T5SmallDecoderInputEmbedDim
      T5SmallFFNDim
      Float ->
    T5Small device dataType

-- | dropout_rate = 0.1
t5SmallDropoutP :: Float
t5SmallDropoutP = 0.1

-- | layer_norm_epsilon = 1e-06
t5SmallEps :: Float
t5SmallEps = 1e-6

type T5SmallAttentionMask device dataType inputSeqSize = Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

t5SmallAttentionMask ::
  forall device dataType inputSeqSize.
  T5SmallAttentionMask device dataType inputSeqSize
t5SmallAttentionMask = undefined

type T5SmallDecoderAttentionMask device dataType decoderInputSeqSize = Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])

t5SmallDecoderAttentionMask ::
  forall device dataType decoderInputSeqSize.
  T5SmallDecoderAttentionMask device dataType decoderInputSeqSize
t5SmallDecoderAttentionMask = undefined

type T5SmallCrossAttentionMask device dataType inputSeqSize decoderInputSeqSize = Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

t5SmallCrossAttentionMask ::
  forall device dataType inputSeqSize decoderInputSeqSize.
  T5SmallCrossAttentionMask device dataType inputSeqSize decoderInputSeqSize
t5SmallCrossAttentionMask = undefined

type T5SmallInput device dataType batchSize inputSeqSize = Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize), T5SmallInputEmbedDim])

type T5SmallDecoderInput device dataType batchSize decoderInputSeqSize = Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), T5SmallDecoderInputEmbedDim])

type T5SmallDecoderOutput device dataType batchSize decoderInputSeqSize = T5SmallDecoderInput device dataType batchSize decoderInputSeqSize

loadT5Small :: IO [(String, [Dim String Integer])]
loadT5Small = do
  let filePath = "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
  iValue <- pickleLoad filePath
  case iValue of
    IVGenericDict xs -> go xs
    _ -> fail "iValue is not a state dictionary."
  where
    go [] = pure []
    go ((IVString s, IVTensor (Torch.Tensor.Unsafe t)) : xs) =
      let t' = UnsafeTensor @ 'Independent @( 'Layout 'Dense) @( 'Device CPU) @ 'UncheckedDataType @ 'UncheckedShape t
       in ((s, shape t') :) <$> go xs
    go ((_, IVTensor _) : _) = fail "iValue is not a string"
    go ((IVString _, _) : _) = fail "iValue is not a tensor"

t5SmallFromPretrained :: FilePath -> IO (T5Small ( 'Device 'CPU) ( 'DataType 'Float))
t5SmallFromPretrained filePath = do
  iValue <- pickleLoad filePath
  stateDict <- case iValue of
    IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  seqToSeq <- do
    encoder <- do
      stack <- do
        undefined
      let layerNorm = LayerNorm (UnsafeTensor undefined) undefined
      let dropout = initialize t5SmallDropoutP
      pure $ TransformerEncoder stack layerNorm dropout
    decoder <- undefined
    pure $ SequenceToSequenceTransformer encoder decoder
  pure $ T5Small seqToSeq
  where
    go [] = pure []
    go ((IVString s, IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, IVTensor _) : _) = fail "iValue is not a string"
    go ((IVString _, _) : _) = fail "iValue is not a tensor"

-- forwardT5Small ::
--   forall device dataType (batchSize :: Nat) (inputSeqSize :: Nat) (decoderInputSeqSize :: Nat) .
--   ( KnownNat batchSize,
--     KnownNat inputSeqSize,
--     KnownNat decoderInputSeqSize,
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim]),
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim])
--   ) =>
--   T5Small device dataType ->
--   T5SmallInput device dataType batchSize inputSeqSize ->
--   T5SmallDecoderInput device dataType batchSize decoderInputSeqSize ->
--   T5SmallAttentionMask device dataType inputSeqSize ->
--   T5SmallDecoderAttentionMask device dataType decoderInputSeqSize ->
--   T5SmallCrossAttentionMask device dataType inputSeqSize decoderInputSeqSize ->
--   Generator device ->
--   ( T5SmallDecoderOutput device dataType batchSize decoderInputSeqSize
--   , Generator device
--   )
-- forwardT5Small model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- forwardT5Small' ::
--   forall device dataType .
--   T5Small device dataType ->
--   Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallDecoderInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Generator device ->
--   ( Tensor 'Dependent ( 'Layout 'Dense) device dataType 'UncheckedShape
--   , Generator device
--   )
-- forwardT5Small' model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- forwardT5Small'' ::
--   T5Small 'UncheckedDevice 'UncheckedDataType ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallDecoderInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Generator 'UncheckedDevice ->
--   ( Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType 'UncheckedShape
--   , Generator 'UncheckedDevice
--   )
-- forwardT5Small'' model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- forwardT5Small''' ::
--   forall (batchSize :: Nat) (inputSeqSize :: Nat) (decoderInputSeqSize :: Nat) .
--   ( KnownNat batchSize,
--     KnownNat inputSeqSize,
--     KnownNat decoderInputSeqSize,
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim]),
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim])
--   ) =>
--   T5Small 'UncheckedDevice 'UncheckedDataType ->
--   T5SmallInput 'UncheckedDevice 'UncheckedDataType batchSize inputSeqSize ->
--   T5SmallDecoderInput 'UncheckedDevice 'UncheckedDataType batchSize decoderInputSeqSize ->
--   T5SmallAttentionMask 'UncheckedDevice 'UncheckedDataType inputSeqSize ->
--   T5SmallDecoderAttentionMask 'UncheckedDevice 'UncheckedDataType decoderInputSeqSize ->
--   T5SmallCrossAttentionMask 'UncheckedDevice 'UncheckedDataType inputSeqSize decoderInputSeqSize ->
--   Generator 'UncheckedDevice ->
--   ( T5SmallDecoderOutput 'UncheckedDevice 'UncheckedDataType batchSize decoderInputSeqSize
--   , Generator 'UncheckedDevice
--   )
-- forwardT5Small''' model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)
