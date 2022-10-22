{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module Torch.Typed.NN.Transformer where

import Control.Monad
import Data.Proxy
import GHC.Generics
import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import Torch.NN (HasForward (..))
import qualified Torch.NN as A
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear, log)
import Torch.Typed.NN.Dropout
import Torch.Typed.NN.Linear
import Torch.Typed.NN.Normalization
import Torch.Typed.NN.Sparse
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (cos, exp, sin)

residual f g x = f x >>= (\x' -> g (x `add` x'))

--------------------------------------------------------------------------------
-- Relation-Aware Multi-Headed Attention Layer
--------------------------------------------------------------------------------

data
  MultiheadAttentionSpec
    (embedDim :: Nat)
    (kEmbedDim :: Nat)
    (vEmbedDim :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttentionSpec ::
    -- | spec for dropout
    DropoutSpec ->
    MultiheadAttentionSpec embedDim kEmbedDim vEmbedDim numHeads dtype device
  deriving (Show, Eq)

data
  MultiheadAttention
    (embedDim :: Nat)
    (kEmbedDim :: Nat)
    (vEmbedDim :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttention ::
    { -- | in-projection for query
      mhaQInProj :: Linear embedDim embedDim dtype device,
      -- | in-projection for key
      mhaKInProj :: Linear kEmbedDim embedDim dtype device,
      -- | in-projection for value
      mhaVInProj :: Linear vEmbedDim embedDim dtype device,
      -- | out-projection
      mhaOutProj :: Linear embedDim embedDim dtype device,
      -- | dropout
      mhaDropout :: Dropout
    } ->
    MultiheadAttention embedDim kEmbedDim vEmbedDim numHeads dtype device
  deriving (Show, Generic, Parameterized)

multiheadAttention ::
  forall embedDim kEmbedDim vEmbedDim numHeads seqLen seqLen' batchSize headDim dtype device.
  ( 1 <= numHeads,
    embedDim ~ (headDim * numHeads),
    All KnownNat '[embedDim, kEmbedDim, vEmbedDim, numHeads, seqLen, seqLen', batchSize, headDim],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | multi-head attention model ADT
  MultiheadAttention embedDim kEmbedDim vEmbedDim numHeads dtype device ->
  -- | switch between training mode and evaluation mode (turns random dropout on and off)
  Bool ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen]) ->
  -- | optional key padding mask
  Maybe (Tensor device 'D.Bool '[batchSize, seqLen]) ->
  -- | optional key relations
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
  -- | optional value relations
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
  -- | query representation
  Tensor device dtype '[batchSize, seqLen', embedDim] ->
  -- | key representation
  Tensor device dtype '[batchSize, seqLen, kEmbedDim] ->
  -- | value representation
  Tensor device dtype '[batchSize, seqLen, vEmbedDim] ->
  -- | attention and attention averaged over heads
  IO
    ( Tensor device dtype '[batchSize, seqLen', embedDim],
      Tensor device dtype '[batchSize, seqLen', seqLen]
    )
multiheadAttention MultiheadAttention {..} train attentionMask keyPaddingMask keyRelations valueRelations query key value = do
  weights <-
    dropoutForward mhaDropout train
      . softmax @3
      . _maskKeyPaddings
      . _maskAttention
      $ _attentionWeights
  pure (_attention weights, averageOverHeads weights)
  where
    _attentionWeights =
      let scaling = Prelude.sqrt . fromIntegral $ natValI @headDim :: Double
          q = reshape' . divScalar scaling . forward mhaQInProj $ query
          k = reshape' . forward mhaKInProj $ key
          weights = matmul q (transpose @2 @3 k)
          weights' = case keyRelations of
            Nothing -> weights
            Just kr -> weights `add` transpose @1 @2 ((transpose @1 @2 q) `matmul` (transpose @2 @3 kr))
       in weights'
    _maskAttention attentionWeights =
      case attentionMask of
        Nothing -> attentionWeights
        Just am -> attentionWeights `add` unsqueeze @1 am
    _maskKeyPaddings attentionWeights =
      case keyPaddingMask of
        Nothing -> attentionWeights
        Just kpm ->
          let keyPaddingMask' = unsqueeze @2 . unsqueeze @1 $ kpm
           in maskedFill keyPaddingMask' (-1 / 0 :: Double) attentionWeights
    _attention attentionWeights =
      let v = reshape' . forward mhaVInProj $ value
          attention = transpose @1 @2 $ matmul attentionWeights v
          attention' = case valueRelations of
            Nothing -> attention
            Just vr -> attention `add` (matmul (transpose @1 @2 attentionWeights) vr)
       in forward mhaOutProj . reshape @'[batchSize, seqLen', embedDim] $ attention'
    averageOverHeads =
      let numHeads' = natValI @numHeads
       in divScalar numHeads' . sumDim @1
    reshape' ::
      forall seqLen''.
      KnownNat seqLen'' =>
      Tensor device dtype '[batchSize, seqLen'', embedDim] ->
      Tensor device dtype '[batchSize, numHeads, seqLen'', headDim]
    reshape' = transpose @1 @2 . reshape @'[batchSize, seqLen'', numHeads, headDim]

instance
  ( All KnownNat '[embedDim, kEmbedDim, vEmbedDim, numHeads],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (MultiheadAttentionSpec embedDim kEmbedDim vEmbedDim numHeads dtype device)
    (MultiheadAttention embedDim kEmbedDim vEmbedDim numHeads dtype device)
  where
  sample (MultiheadAttentionSpec mhaDropoutSpec) =
    MultiheadAttention
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample mhaDropoutSpec

--------------------------------------------------------------------------------
-- Transformer MLP Layer
--------------------------------------------------------------------------------

data
  TransformerMLPSpec
    (embedDim :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLPSpec ::
    forall embedDim ffnDim dtype device.
    { -- | spec for relu dropout
      dropout0Spec :: DropoutSpec,
      -- | spec for other dropout
      dropout1Spec :: DropoutSpec,
      -- | epsilon for layer norm
      epsSpec :: Double
    } ->
    TransformerMLPSpec embedDim ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerMLP
    (embedDim :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLP ::
    forall embedDim ffnDim dtype device.
    { -- | first fully connected layer
      linear0 :: Linear embedDim ffnDim dtype device,
      -- | second fully connected layer
      linear1 :: Linear ffnDim embedDim dtype device,
      -- | relu dropout
      dropout0 :: Dropout,
      -- | other dropout
      dropout1 :: Dropout,
      -- | layer norm
      ln :: LayerNorm '[embedDim] dtype device
    } ->
    TransformerMLP embedDim ffnDim dtype device
  deriving (Show, Generic, Parameterized)

transformerMLP ::
  forall embedDim ffnDim seqLen batchSize dtype device.
  ( BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    KnownNat embedDim,
    IsSuffixOf '[embedDim] '[seqLen, batchSize, embedDim]
  ) =>
  -- | MLP model ADT for transformer
  TransformerMLP embedDim ffnDim dtype device ->
  -- | switch between training mode and evaluation mode (turns random dropout on and off)
  Bool ->
  Tensor device dtype '[seqLen, batchSize, embedDim] -> -- input
  IO (Tensor device dtype '[seqLen, batchSize, embedDim]) -- output
transformerMLP TransformerMLP {..} train input =
  residual f (pure . forward ln) input
  where
    f x =
      dropoutForward dropout1 train
        . forward linear1
        =<< dropoutForward dropout0 train
          . relu
          . forward linear0
        =<< pure x

instance
  ( All KnownNat '[embedDim, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerMLPSpec embedDim ffnDim dtype device)
    (TransformerMLP embedDim ffnDim dtype device)
  where
  sample TransformerMLPSpec {..} =
    TransformerMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample dropout0Spec
      <*> A.sample dropout1Spec
      <*> A.sample (LayerNormSpec epsSpec)

--------------------------------------------------------------------------------
-- Relation-Aware Transformer Layer
--------------------------------------------------------------------------------

data
  TransformerLayerSpec
    (embedDim :: Nat)
    (kEmbedDim :: Nat)
    (vEmbedDim :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayerSpec ::
    forall embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device.
    { mhaSpec :: MultiheadAttentionSpec embedDim kEmbedDim vEmbedDim numHeads dtype device,
      attnDropoutSpec :: DropoutSpec,
      epsSpec' :: Double,
      mlpSpec :: TransformerMLPSpec embedDim ffnDim dtype device
    } ->
    TransformerLayerSpec embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerLayer
    (embedDim :: Nat)
    (kEmbedDim :: Nat)
    (vEmbedDim :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayer ::
    forall embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device.
    { -- | multi-head attention
      transformerLayer_mha :: MultiheadAttention embedDim kEmbedDim vEmbedDim numHeads dtype device,
      -- | dropout
      transformerLayer_attnDropout :: Dropout,
      -- | layer norm
      transformerLayer_ln :: LayerNorm '[embedDim] dtype device,
      -- | MLP
      transformerLayer_mlp :: TransformerMLP embedDim ffnDim dtype device
    } ->
    TransformerLayer embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device
  deriving (Show, Generic, Parameterized)

transformerLayer ::
  forall (numHeads :: Nat) (ffnDim :: Nat) (embedDim :: Nat) (kEmbedDim :: Nat) (vEmbedDim :: Nat) (headDim :: Nat) (seqLen :: Nat) (seqLen' :: Nat) (batchSize :: Nat) dtype device.
  ( 1 <= numHeads,
    embedDim ~ (headDim * numHeads),
    All KnownNat '[embedDim, kEmbedDim, vEmbedDim, numHeads, seqLen, seqLen', batchSize, headDim],
    IsSuffixOf '[embedDim] '[batchSize, seqLen', embedDim],
    KnownDType dtype,
    dtype ~ SumDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | transformer layer model ADT
  TransformerLayer embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device ->
  -- | switch between training mode and evaluation mode (turns random dropout on and off)
  Bool ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen]) ->
  -- | optional key padding mask
  Maybe (Tensor device 'D.Bool '[batchSize, seqLen]) ->
  -- | optional key relations
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
  -- | optional value relations
  Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
  -- | query representation
  Tensor device dtype '[batchSize, seqLen', embedDim] ->
  -- | key representation
  Tensor device dtype '[batchSize, seqLen, kEmbedDim] ->
  -- | value representation
  Tensor device dtype '[batchSize, seqLen, vEmbedDim] ->
  -- | transformer layer output representation
  IO (Tensor device dtype '[batchSize, seqLen', embedDim])
transformerLayer TransformerLayer {..} train attentionMask keyPaddingMask keyRelations valueRelations query key value =
  let f query' =
        multiheadAttention transformerLayer_mha train attentionMask keyPaddingMask keyRelations valueRelations query' key value
          >>= dropoutForward transformerLayer_attnDropout train . fst
   in residual f (pure . forward transformerLayer_ln) query >>= transformerMLP transformerLayer_mlp train

instance
  ( All KnownNat '[embedDim, kEmbedDim, vEmbedDim, numHeads, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerLayerSpec embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device)
    (TransformerLayer embedDim kEmbedDim vEmbedDim numHeads ffnDim dtype device)
  where
  sample TransformerLayerSpec {..} =
    TransformerLayer
      <$> A.sample mhaSpec
      <*> A.sample attnDropoutSpec
      <*> A.sample (LayerNormSpec epsSpec')
      <*> A.sample mlpSpec

--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data
  TransformerLMSpec
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (numEmbeds :: Nat)
    (embedDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLMSpec ::
    forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device.
    { -- | dropout spec
      lmDropoutSpec :: DropoutSpec,
      -- | spec for each and every transformer layer
      lmLayerSpec :: TransformerLayerSpec embedDim embedDim embedDim numHeads ffnDim dtype device
    } ->
    TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
  deriving (Show, Eq)

data
  TransformerLM
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (numEmbeds :: Nat)
    (embedDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLM ::
    forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device.
    { -- | token embedding
      tEmbedding :: Embedding ('Just paddingIdx) numEmbeds embedDim 'Learned dtype device,
      -- | positional embedding
      tPosEmbedding :: Embedding 'Nothing 2048 embedDim 'Constant dtype device,
      -- | transformer dropout
      tDropout :: Dropout,
      -- | transformer layers
      tLayers :: HList (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device)),
      -- | final output projection
      tProj :: Linear embedDim numEmbeds dtype device
    } ->
    TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
  deriving (Generic)

deriving instance
  ( Show
      ( HList
          ( HReplicateR
              numAttnLayers
              ( TransformerLayer
                  embedDim
                  embedDim
                  embedDim
                  numHeads
                  ffnDim
                  dtype
                  device
              )
          )
      )
  ) =>
  Show (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)

instance
  ( layers
      ~ ( HReplicateR
            numAttnLayers
            ( TransformerLayer
                embedDim
                embedDim
                embedDim
                numHeads
                ffnDim
                dtype
                device
            )
        ),
    Parameterized
      ( HList
          layers
      ),
    HAppendFD
      (Parameters (HList layers))
      '[ Parameter device dtype '[numEmbeds, embedDim],
         Parameter device dtype '[numEmbeds]
       ]
      ( Parameters (HList layers)
          ++ '[ Parameter device dtype '[numEmbeds, embedDim],
                Parameter device dtype '[numEmbeds]
              ]
      )
  ) =>
  Parameterized (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)

data
  FoldLayers
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) = FoldLayers
  { -- | switch between training mode and evaluation mode (turns random dropout on and off)
    flTrain :: Bool,
    -- | optional attention mask
    flAttentionMask :: Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]),
    -- | optional key padding mask
    flKeyPaddingMask :: Maybe (Tensor device 'D.Bool '[batchSize, seqLen])
  }

instance
  ( 1 <= numHeads,
    embedDim ~ (headDim * numHeads),
    All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim],
    IsSuffixOf '[embedDim] '[batchSize, seqLen, embedDim],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Apply'
    (FoldLayers batchSize seqLen dtype device)
    ( TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device,
      IO (Tensor device dtype '[batchSize, seqLen, embedDim])
    )
    (IO (Tensor device dtype '[batchSize, seqLen, embedDim]))
  where
  apply' FoldLayers {..} (layer, mx) = mx >>= \x -> transformerLayer layer flTrain flAttentionMask flKeyPaddingMask Nothing Nothing x x x

transformerLM ::
  forall
    numAttnLayers
    numHeads
    ffnDim
    paddingIdx
    numEmbeds
    embedDim
    seqLen
    batchSize
    dtype
    device.
  ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize],
    paddingIdx + 1 <= numEmbeds,
    1 <= seqLen,
    HFoldrM
      IO
      (FoldLayers batchSize seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, embedDim])
      (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device))
      (Tensor device dtype '[batchSize, seqLen, embedDim]),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device ->
  Bool ->
  Tensor device 'D.Int64 '[batchSize, seqLen] ->
  IO (Tensor device dtype '[batchSize, seqLen, numEmbeds])
transformerLM TransformerLM {..} train xTokens = do
  let x = embed tEmbedding xTokens
      positions =
        expand @'[batchSize, seqLen, embedDim] True
          . embed tPosEmbedding
          . Torch.Typed.Tensor.toDType @D.Int64
          . linspace @seqLen (0 :: Int)
          $ natValI @(seqLen - 1)
  x' <- dropoutForward tDropout train (x `add` positions)
  let attentionMask =
        unsqueeze @0
          . Torch.Typed.Tensor.toDType @D.Bool
          . triu 1
          $ ones @'[seqLen, seqLen] @D.Int8 @device
      attentionMask' =
        pure . maskedFill attentionMask (-1 / 0 :: Double) $
          zeros @'[batchSize, seqLen, seqLen] @dtype @device
  let keyPaddingMask = pure $ xTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[])
  y <- hfoldrM (FoldLayers train attentionMask' keyPaddingMask) x' tLayers
  return $ forward tProj y

instance
  ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize],
    paddingIdx + 1 <= numEmbeds,
    1 <= seqLen,
    HFoldrM
      IO
      (FoldLayers batchSize seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, embedDim])
      (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device))
      (Tensor device dtype '[batchSize, seqLen, embedDim]),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  HasForward (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device) (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, numEmbeds])
  where
  forward model input = unsafePerformIO $ transformerLM model False input
  forwardStoch model input = transformerLM model True input

sinusoidal ::
  forall numEmbeds embedDim device.
  ( All KnownNat '[numEmbeds, embedDim],
    1 <= numEmbeds,
    1 <= Div embedDim 2,
    (Div embedDim 2 * 2) ~ embedDim,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  Tensor device 'D.Float '[numEmbeds, embedDim]
sinusoidal =
  let positions =
        unsqueeze @1
          . linspace @numEmbeds (0 :: Int)
          $ natValI @(numEmbeds - 1)
      scalingFactors =
        exp
          . mulScalar (- log (10000 :: Double) / (fromInteger . natVal $ Proxy @(Div embedDim 2)))
          . linspace @(Div embedDim 2) (0 :: Int)
          $ natValI @((Div embedDim 2) - 1)
      radians = mul positions scalingFactors
      weights = stack @2 (sin radians :. cos radians :. HNil)
   in reshape weights

instance
  ( paddingIdx <= numEmbeds,
    1 <= numEmbeds - paddingIdx,
    1 <= Div embedDim 2,
    (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds,
    (Div embedDim 2 * 2) ~ embedDim,
    All KnownNat '[ffnDim, paddingIdx, numEmbeds, embedDim],
    HReplicate numAttnLayers (TransformerLayerSpec embedDim embedDim embedDim numHeads ffnDim dtype device),
    A.Randomizable
      (HList (HReplicateR numAttnLayers (TransformerLayerSpec embedDim embedDim embedDim numHeads ffnDim dtype device)))
      (HList (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device))),
    KnownDType dtype,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  A.Randomizable
    (TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)
    (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)
  where
  sample TransformerLMSpec {..} =
    TransformerLM
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample lmDropoutSpec
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample LinearSpec
