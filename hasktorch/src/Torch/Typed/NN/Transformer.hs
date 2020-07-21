{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
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
import qualified Torch.NN as A
import Torch.NN (HasForward(..))
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear, log)
import Torch.Typed.Tensor
import Torch.Typed.NN.Dropout
import Torch.Typed.NN.Linear
import Torch.Typed.NN.Normalization
import Torch.Typed.NN.Sparse
import Prelude hiding (cos, exp, sin)

--------------------------------------------------------------------------------
-- Relation-Aware Multi-Headed Attention Layer
--------------------------------------------------------------------------------

data
  MultiheadAttentionSpec
    (embedDim :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  MultiheadAttentionSpec
    :: { mhaDropoutSpec :: DropoutSpec }
    -> MultiheadAttentionSpec embedDim numHeads dtype device
  deriving (Show, Eq)

data
  MultiheadAttention
    (embedDim :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  MultiheadAttention
    :: { mhaInProj  :: Linear embedDim (embedDim * 3) dtype device
       , mhaOutProj :: Linear embedDim embedDim       dtype device
       , mhaDropout :: Dropout
       }
    -> MultiheadAttention embedDim numHeads dtype device
  deriving (Show, Generic)

multiheadAttention
  :: forall embedDim numHeads seqLen batchSize headDim dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     , KnownDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     , MatMulDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , dtype ~ SumDType dtype
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => MultiheadAttention embedDim numHeads dtype device
  -> Bool
  -> Tensor device dtype '[batchSize, seqLen, seqLen]
  -> Tensor device 'D.Bool '[batchSize, seqLen]
  -> Maybe (Tensor device dtype '[batchSize, seqLen, seqLen, headDim])
  -> Maybe (Tensor device dtype '[batchSize, seqLen, seqLen, headDim])
  -> Tensor device dtype '[batchSize, seqLen, embedDim]
  -> IO ( Tensor device dtype '[batchSize, seqLen, embedDim]
        , Tensor device dtype '[batchSize, seqLen, seqLen]
        )
multiheadAttention MultiheadAttention {..} train attentionMask keyPaddingMask maybeRelationsK maybeRelationsV x = do
  let q :. k :. v :. HNil = chunk @3 @2 . forward mhaInProj $ x
      q' = reshape' q
      k' = reshape' k
      v' = reshape' v
  coefficients <- weightCoefficients q' k'
  return (attention coefficients v', averageOverHeads coefficients)
  where
    weightCoefficients q k =
      dropoutForward mhaDropout train
        . softmax @3
        . maskKeyPaddings
        . maskAttention
        . divScalar scaling
        . ap maybe add (matmul q . transpose @2 @3 $ k)
        . fmap (transpose @1 @2 . matmul (transpose @1 @2 q) . transpose @2 @3)
        $ maybeRelationsK
    attention coefficients v =
      forward mhaOutProj
        . reshape @'[batchSize, seqLen, embedDim]
        . ap maybe add (transpose @1 @2 . matmul coefficients $ v)
        . fmap (matmul (transpose @1 @2 coefficients))
        $ maybeRelationsV
    averageOverHeads =
      let numHeads' = natValI @numHeads
       in divScalar numHeads' . sumDim @1
    maskAttention = add (unsqueeze @1 attentionMask)
    maskKeyPaddings =
      let keyPaddingMask' = unsqueeze @2 . unsqueeze @1 $ keyPaddingMask
       in maskedFill keyPaddingMask' (-1 / 0 :: Double)
    reshape' = transpose @1 @2 . reshape @'[batchSize, seqLen, numHeads, headDim]
    scaling = Prelude.sqrt . fromIntegral $ natValI @headDim :: Double

instance ( All KnownNat '[embedDim, numHeads]
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (MultiheadAttentionSpec embedDim numHeads dtype device)
                    (MultiheadAttention     embedDim numHeads dtype device)
 where
  sample MultiheadAttentionSpec {..} =
    MultiheadAttention
      <$> A.sample LinearSpec
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
    (device :: (D.DeviceType, Nat)) where
  TransformerMLPSpec
    :: forall embedDim ffnDim dtype device
     . { dropout0Spec :: DropoutSpec
       , dropout1Spec :: DropoutSpec
       }
    -> TransformerMLPSpec embedDim ffnDim dtype device
  deriving Show

data
  TransformerMLP
    (embedDim :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  TransformerMLP
    :: forall embedDim ffnDim dtype device
     . { linear0     :: Linear embedDim ffnDim dtype device
       , linear1     :: Linear ffnDim embedDim dtype device
       , dropout0    :: Dropout
       , dropout1    :: Dropout
       }
    -> TransformerMLP embedDim ffnDim dtype device
 deriving (Show, Generic)

transformerMLP
  :: forall embedDim ffnDim seqLen batchSize dtype device
   . StandardFloatingPointDTypeValidation device dtype
  => TransformerMLP embedDim ffnDim dtype device
  -> Bool
  -> Tensor device dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor device dtype '[seqLen, batchSize, embedDim])
transformerMLP TransformerMLP {..} train input =
  dropoutForward dropout1 train
    .   relu
    .   forward linear1
    =<< dropoutForward dropout0 train
    .   relu
    .   forward linear0
    =<< pure input

instance ( All KnownNat '[embedDim, ffnDim]
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (TransformerMLPSpec embedDim ffnDim dtype device)
                    (TransformerMLP     embedDim ffnDim dtype device)
 where
  sample TransformerMLPSpec {..} =
    TransformerMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample dropout0Spec
      <*> A.sample dropout1Spec

--------------------------------------------------------------------------------
-- Relation-Aware Transformer Layer
--------------------------------------------------------------------------------

data
  TransformerLayerSpec
    (embedDim :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  TransformerLayerSpec
    :: forall embedDim numHeads ffnDim dtype device
     . { mhaSpec         :: MultiheadAttentionSpec embedDim numHeads dtype device
       , attnDropoutSpec :: DropoutSpec
       , epsSpec         :: Double
       , mlpSpec         :: TransformerMLPSpec embedDim ffnDim dtype device
       }
    -> TransformerLayerSpec embedDim numHeads ffnDim dtype device
  deriving (Show)

data
  TransformerLayer
    (embedDim :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  TransformerLayer
    :: forall embedDim numHeads ffnDim dtype device
     . { transformerLayer_mha         :: MultiheadAttention embedDim numHeads dtype device
       , transformerLayer_attnDropout :: Dropout
       , transformerLayer_ln0         :: LayerNorm '[embedDim] dtype device
       , transformerLayer_ln1         :: LayerNorm '[embedDim] dtype device
       , transformerLayer_mlp         :: TransformerMLP embedDim ffnDim dtype device
       }
    -> TransformerLayer embedDim numHeads ffnDim dtype device
  deriving (Show, Generic)

transformerLayer
  :: forall numHeads ffnDim embedDim headDim seqLen batchSize dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     , EndsWith '[batchSize, seqLen, embedDim] '[embedDim]
     , KnownDType dtype
     , dtype ~ SumDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     , MatMulDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => TransformerLayer embedDim numHeads ffnDim dtype device
  -> Bool
  -> Tensor device dtype '[batchSize, seqLen, seqLen]
  -> Tensor device 'D.Bool '[batchSize, seqLen]
  -> Maybe (Tensor device dtype '[batchSize, seqLen, seqLen, headDim])
  -> Maybe (Tensor device dtype '[batchSize, seqLen, seqLen, headDim])
  -> Tensor device dtype '[batchSize, seqLen, embedDim]
  -> IO (Tensor device dtype '[batchSize, seqLen, embedDim])
transformerLayer TransformerLayer {..} train attentionMask keyPaddingMask maybeRelationsK maybeRelationsV x = do
  (z, _) <- multiheadAttention transformerLayer_mha train attentionMask keyPaddingMask maybeRelationsK maybeRelationsV x
  z' <- dropoutForward transformerLayer_attnDropout train z
  let y = forward transformerLayer_ln0 (x `add` z')
  y' <- transformerMLP transformerLayer_mlp train y
  return $ forward transformerLayer_ln1 (y `add` y')

instance ( All KnownNat '[embedDim, numHeads, ffnDim]
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (TransformerLayerSpec embedDim numHeads ffnDim dtype device)
                    (TransformerLayer     embedDim numHeads ffnDim dtype device)
 where
  sample TransformerLayerSpec {..} =
    TransformerLayer
      <$> A.sample mhaSpec
      <*> A.sample attnDropoutSpec
      <*> A.sample (LayerNormSpec epsSpec)
      <*> A.sample (LayerNormSpec epsSpec)
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
    (device :: (D.DeviceType, Nat)) where
  TransformerLMSpec
    :: forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
     . { lmDropoutSpec :: DropoutSpec
       , lmLayerSpec   :: TransformerLayerSpec embedDim numHeads ffnDim dtype device
       }
    -> TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
  deriving (Show)

data
  TransformerLM
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (numEmbeds :: Nat)
    (embedDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  TransformerLM
    :: forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
     . { tEmbedding    :: Embedding ('Just paddingIdx) numEmbeds embedDim 'Learned dtype device
       , tPosEmbedding :: Embedding 'Nothing           2048      embedDim 'Constant dtype device
       , tDropout      :: Dropout
       , tLayers       :: HList (HReplicateR numAttnLayers (TransformerLayer embedDim numHeads ffnDim dtype device))
       , tProj         :: Linear embedDim numEmbeds dtype device
       }
    -> TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
  deriving (Generic)

data
  FoldLayers
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = FoldLayers
      { flTrain :: Bool
      , flAttentionMask :: Tensor device dtype '[batchSize, seqLen, seqLen]
      , flKeyPaddingMask :: Tensor device 'D.Bool '[batchSize, seqLen]
      }

instance
  ( 1 <= numHeads
  , embedDim ~ (headDim * numHeads)
  , Mod (embedDim * 3) 3 ~ 0
  , Div (embedDim * 3) 3 ~ embedDim
  , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
  , EndsWith '[batchSize, seqLen, embedDim] '[embedDim]
  , KnownDType dtype
  , dtype ~ SumDType dtype
  , StandardFloatingPointDTypeValidation device dtype
  , MatMulDTypeIsValid device dtype
  , BasicArithmeticDTypeIsValid device dtype
  , SumDTypeIsValid device dtype
  , KnownDevice device
  ) => Apply' (FoldLayers batchSize seqLen dtype device) (TransformerLayer embedDim numHeads ffnDim dtype device, IO (Tensor device dtype '[batchSize, seqLen, embedDim])) (IO (Tensor device dtype '[batchSize, seqLen, embedDim])) where
  apply' FoldLayers {..} (layer, mx) = mx >>= transformerLayer layer flTrain flAttentionMask flKeyPaddingMask Nothing Nothing

transformerLM
  :: forall
       numAttnLayers
       numHeads
       ffnDim
       paddingIdx
       numEmbeds
       embedDim
       seqLen
       batchSize
       dtype
       device
   . ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
     , paddingIdx + 1 <= numEmbeds
     , 1 <= seqLen
     , HFoldrM
         IO
         (FoldLayers batchSize seqLen dtype device)
         (Tensor device dtype '[batchSize, seqLen, embedDim])
         (HReplicateR numAttnLayers (TransformerLayer embedDim numHeads ffnDim dtype device))
         (Tensor device dtype '[batchSize, seqLen, embedDim])
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDType dtype
     , KnownDevice device
     )
  => TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device
  -> Bool
  -> Tensor device 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor device dtype '[batchSize, seqLen, numEmbeds])
transformerLM TransformerLM {..} train xTokens = do
  let x         = embed tEmbedding xTokens
      positions = expand @'[batchSize, seqLen, embedDim] True
                    . embed tPosEmbedding
                    . Torch.Typed.Tensor.toDType @D.Int64
                    . linspace @seqLen (0 :: Int)
                    $ natValI @(seqLen - 1)
  x' <- dropoutForward tDropout train (x `add` positions)
  let attentionMask = unsqueeze @0
                        . Torch.Typed.Tensor.toDType @D.Bool
                        . triu 1
                        $ ones @'[seqLen, seqLen] @D.Int8 @device
      attentionMask' = maskedFill attentionMask (-1 / 0 :: Double)
                         $ zeros @'[batchSize, seqLen, seqLen] @dtype @device
  let keyPaddingMask = xTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[])
  y <- hfoldrM (FoldLayers train attentionMask' keyPaddingMask) x' tLayers
  return $ forward tProj y

instance
  ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
  , paddingIdx + 1 <= numEmbeds
  , 1 <= seqLen
  , HFoldrM
      IO
      (FoldLayers batchSize seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, embedDim])
      (HReplicateR numAttnLayers (TransformerLayer embedDim numHeads ffnDim dtype device))
      (Tensor device dtype '[batchSize, seqLen, embedDim])
  , BasicArithmeticDTypeIsValid device dtype
  , ComparisonDTypeIsValid device dtype
  , ComparisonDTypeIsValid device 'D.Int64
  , KnownDType dtype
  , KnownDevice device
  ) => HasForward (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device) (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, numEmbeds]) where
  forward model input = unsafePerformIO $ transformerLM model False input
  forwardStoch model input = transformerLM model True input

sinusoidal
  :: forall numEmbeds embedDim device
   . ( All KnownNat '[numEmbeds, embedDim]
     , 1 <= numEmbeds
     , 1 <= Div embedDim 2
     , (Div embedDim 2 * 2) ~ embedDim
     , StandardFloatingPointDTypeValidation device 'D.Float
     , BasicArithmeticDTypeIsValid device 'D.Float
     , KnownDevice device
     )
  => Tensor device 'D.Float '[numEmbeds, embedDim]
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
  in  reshape weights

instance
  ( paddingIdx <= numEmbeds
  , 1 <= numEmbeds - paddingIdx
  , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
  , (Div embedDim 2 * 2) ~ embedDim
  , All KnownNat '[ffnDim, paddingIdx, numEmbeds, embedDim]
  , HReplicate numAttnLayers (TransformerLayerSpec embedDim numHeads ffnDim dtype device)
  , A.Randomizable (HList (HReplicateR numAttnLayers (TransformerLayerSpec embedDim numHeads ffnDim dtype device)))
                   (HList (HReplicateR numAttnLayers (TransformerLayer     embedDim numHeads ffnDim dtype device)))
  , KnownDType dtype
  , RandDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device 'D.Float
  , BasicArithmeticDTypeIsValid device 'D.Float
  , KnownDevice device
  ) => A.Randomizable (TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)
                      (TransformerLM     numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype device)
 where
  sample TransformerLMSpec {..} =
    TransformerLM
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @( 'Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @ 'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample lmDropoutSpec
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample LinearSpec
