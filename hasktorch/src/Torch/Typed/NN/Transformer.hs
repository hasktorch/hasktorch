{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module Torch.Typed.NN.Transformer where

import           Prelude hiding (exp, sin, cos)
import           Torch.HList
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Functional     hiding ( linear, log )
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional              as D
import qualified Torch.TensorFactories         as D


--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data MultiheadAttentionSpec (embedDim :: Nat) (numHeads :: Nat)
                            (dtype :: D.DType)
                            (device :: (D.DeviceType, Nat))
 where
  MultiheadAttentionSpec
    :: { mhaDropoutSpec :: DropoutSpec }
    -> MultiheadAttentionSpec embedDim numHeads dtype device
 deriving (Show, Eq)

data MultiheadAttention (embedDim :: Nat) (numHeads :: Nat)
                        (dtype :: D.DType)
                        (device :: (D.DeviceType, Nat))
 where
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
     , dtype ~ DTypePromotion dtype (SumDType dtype)
     , StandardFloatingPointDTypeValidation device dtype
     , MatMulDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device (SumDType dtype)
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => MultiheadAttention embedDim numHeads dtype device
  -> Bool
  -> Tensor device 'D.Bool '[seqLen, batchSize]
  -> Tensor device dtype   '[seqLen, batchSize, embedDim]
  -> IO ( Tensor device dtype '[seqLen, batchSize, embedDim]
        , Tensor device dtype '[batchSize, seqLen, seqLen]
        )
multiheadAttention MultiheadAttention {..} train keyPaddingMask input = do
  let q :. k :. v :. HNil = chunk @3 @2 . linear mhaInProj $ input
  attnWeights <-
    Torch.Typed.NN.dropout mhaDropout train
      . softmax @2
      . maskKeyPaddings
      . maskFutureTimesteps
      $ (mul scaling $ f q) `matmul` (transpose @1 @2 $ f k)
  let attn =
        linear mhaOutProj
          . reshape @'[seqLen, batchSize, embedDim]
          . transpose @0 @1
          $ attnWeights `matmul` (f v)
      avgAttnWeights =
        mul (pow (-1 :: Double) (fromInteger . natVal $ Proxy @numHeads) :: Tensor device dtype '[])
          . sumDim @1
          . reshape @'[batchSize, numHeads, seqLen, seqLen]
          $ attnWeights
  return (attn, avgAttnWeights)
 where
  maskFutureTimesteps = maskedFill futureTimestampMask (-1 / 0 :: Double)
  maskKeyPaddings =
    reshape @'[batchSize * numHeads, seqLen, seqLen]
      . maskedFill
          (unsqueeze @2 . unsqueeze @1 . transpose @0 @1 $ keyPaddingMask)
          (-1 / 0 :: Double)
      . reshape @'[batchSize, numHeads, seqLen, seqLen]
  f = transpose @0 @1 . reshape @'[seqLen, batchSize * numHeads, headDim]
  scaling :: Tensor device dtype '[]
  scaling = pow (-1 / 2 :: Double) (fromInteger . natVal $ Proxy @headDim)
  futureTimestampMask =
    Torch.Typed.Tensor.toDType @D.Bool . triu 1 $ ones @'[seqLen, seqLen] @D.Int8 @device

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

data TransformerLMLayerSpec (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
                            (dtype :: D.DType)
                            (device :: (D.DeviceType, Nat))
 where
  TransformerLMLayerSpec
    :: forall embedDim numHeads ffnDim dtype device
     . { mhaSpec         :: MultiheadAttentionSpec embedDim numHeads dtype device
       , attnDropoutSpec :: DropoutSpec
       , epsSpec         :: Double
       , mlpSpec         :: TransformerLMMLPSpec embedDim ffnDim dtype device
       }
    -> TransformerLMLayerSpec embedDim numHeads ffnDim dtype device
 deriving (Show)

data TransformerLMLayer (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
                        (dtype :: D.DType)
                        (device :: (D.DeviceType, Nat))
 where
  TransformerLMLayer
    :: forall embedDim numHeads ffnDim dtype device
     . { mha         :: MultiheadAttention embedDim numHeads dtype device
       , attnDropout :: Dropout
       , ln0         :: LayerNorm '[embedDim] dtype device
       , ln1         :: LayerNorm '[embedDim] dtype device
       , mlp         :: TransformerLMMLP embedDim ffnDim dtype device
       }
    -> TransformerLMLayer embedDim numHeads ffnDim dtype device
 deriving (Show, Generic)

transformerLMLayer
  :: forall numHeads ffnDim embedDim headDim seqLen batchSize dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
     , KnownDType dtype
     , dtype ~ SumDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     , MatMulDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => TransformerLMLayer embedDim numHeads ffnDim dtype device
  -> Bool
  -> Tensor device 'D.Bool '[seqLen, batchSize]
  -> Tensor device dtype   '[seqLen, batchSize, embedDim]
  -> IO (Tensor device dtype '[seqLen, batchSize, embedDim])
transformerLMLayer TransformerLMLayer {..} train keyPaddingMask input = do
  (attn, _) <- multiheadAttention mha train keyPaddingMask input
  x         <- Torch.Typed.NN.dropout attnDropout train attn
  let x' = Torch.Typed.NN.layerNorm ln0 (x `add` input)
  x''       <- transformerLMMLP mlp train x'
  return $ Torch.Typed.NN.layerNorm ln1 (x'' `add` x')

instance ( All KnownNat '[embedDim, numHeads, ffnDim]
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)
                    (TransformerLMLayer     embedDim numHeads ffnDim dtype device)
 where
  sample TransformerLMLayerSpec {..} =
    TransformerLMLayer
      <$> A.sample mhaSpec
      <*> A.sample attnDropoutSpec
      <*> A.sample (LayerNormSpec epsSpec)
      <*> A.sample (LayerNormSpec epsSpec)
      <*> A.sample mlpSpec

data Activation (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  Activation
    :: forall dtype device
     . { unActivation :: forall shape . Tensor device dtype shape -> Tensor device dtype shape }
    -> Activation dtype device

instance Show (Activation dtype device) where
  -- we can't show functions :(
  show _ = mempty

instance {-# OVERLAPS #-} Parameterized (Activation dtype device) '[] where
  flattenParameters _ = HNil
  replaceParameters = const

data TransformerLMMLPSpec (embedDim :: Nat) (ffnDim :: Nat)
                          (dtype :: D.DType)
                          (device :: (D.DeviceType, Nat))
 where
  TransformerLMMLPSpec
    :: forall embedDim ffnDim dtype device
     . { dropout0Spec :: DropoutSpec
       , dropout1Spec :: DropoutSpec
       , activation0Spec :: Activation dtype device
       , activation1Spec :: Activation dtype device
       }
    -> TransformerLMMLPSpec embedDim ffnDim dtype device
 deriving Show

data TransformerLMMLP (embedDim :: Nat) (ffnDim :: Nat)
                      (dtype :: D.DType)
                      (device :: (D.DeviceType, Nat))
 where
  TransformerLMMLP
    :: forall embedDim ffnDim dtype device
     . { linear0     :: Linear embedDim ffnDim dtype device
       , linear1     :: Linear ffnDim embedDim dtype device
       , dropout0    :: Dropout
       , dropout1    :: Dropout
       , activation0 :: Activation dtype device
       , activation1 :: Activation dtype device
       }
    -> TransformerLMMLP embedDim ffnDim dtype device
 deriving (Show, Generic)

transformerLMMLP
  :: forall embedDim ffnDim seqLen batchSize dtype device
   . TransformerLMMLP embedDim ffnDim dtype device
  -> Bool
  -> Tensor device dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor device dtype '[seqLen, batchSize, embedDim])
transformerLMMLP TransformerLMMLP {..} train input =
  Torch.Typed.NN.dropout dropout1 train
    .   unActivation activation1
    .   linear linear1
    =<< Torch.Typed.NN.dropout dropout0 train
    .   unActivation activation0
    .   linear linear0
    =<< pure input

instance ( All KnownNat '[embedDim, ffnDim]
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (TransformerLMMLPSpec embedDim ffnDim dtype device)
                    (TransformerLMMLP     embedDim ffnDim dtype device)
 where
  sample TransformerLMMLPSpec {..} =
    TransformerLMMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample dropout0Spec
      <*> A.sample dropout1Spec
      <*> pure activation0Spec
      <*> pure activation1Spec

data FoldLayers = FoldLayers { foldLayersTrain :: Bool }

instance ( 1 <= numHeads
         , embedDim ~ (headDim * numHeads)
         , Mod (embedDim * 3) 3 ~ 0
         , Div (embedDim * 3) 3 ~ embedDim
         , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
         , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
         , KnownDType dtype
         , dtype ~ SumDType dtype
         , StandardFloatingPointDTypeValidation device dtype
         , MatMulDTypeIsValid device dtype
         , BasicArithmeticDTypeIsValid device dtype
         , SumDTypeIsValid device dtype
         , KnownDevice device
         )
    => Apply
         FoldLayers
         (TransformerLMLayer embedDim numHeads ffnDim dtype device)
         ((Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim]) -> IO (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim]))
 where
  apply FoldLayers {..} layer = \(keyPaddingMask, input) -> do
    output <- transformerLMLayer layer foldLayersTrain keyPaddingMask input
    return (keyPaddingMask, output)

getHidden
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
         FoldLayers
         (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDevice device
     )
  => Embedding ('Just paddingIdx) numEmbeds embedDim 'Learned dtype device
  -> Embedding 'Nothing           2048      embedDim 'Constant dtype device
  -> Dropout
  -> Bool
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
  -> Tensor device 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor device dtype '[seqLen, batchSize, embedDim])
getHidden embedding posEmbedding dropout train layers input = do
  let srcTokens = transpose @0 @1 input
      src       = embed embedding srcTokens
      positions = expand @'[seqLen, batchSize, embedDim] True
                    . unsqueeze @1
                    . embed posEmbedding
                    . Torch.Typed.Tensor.toDType @D.Int64
                    . linspace @seqLen (0 :: Int)
                    $ natValI @(seqLen - 1)
  x <- Torch.Typed.NN.dropout dropout train (src `add` positions)
  let keyPaddingMask = srcTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[])
  (_, x') <- hfoldrM (FoldLayers train) (keyPaddingMask, x) layers
  return x'

data TransformerLMSpec
       (numAttnLayers :: Nat)
       (numHeads :: Nat)
       (ffnDim :: Nat)
       (paddingIdx :: Nat)
       (numEmbeds :: Nat)
       (embedDim :: Nat)
       (seqLen :: Nat)
       (dtype :: D.DType)
       (device :: (D.DeviceType, Nat))
 where
  TransformerLMSpec
    :: forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
     . { lmDropoutSpec :: DropoutSpec
       , lmLayerSpec   :: TransformerLMLayerSpec embedDim numHeads ffnDim dtype device
       }
    -> TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
 deriving (Show)

data TransformerLM
       (numAttnLayers :: Nat)
       (numHeads :: Nat)
       (ffnDim :: Nat)
       (paddingIdx :: Nat)
       (numEmbeds :: Nat)
       (embedDim :: Nat)
       (seqLen :: Nat)
       (dtype :: D.DType)
       (device :: (D.DeviceType, Nat))
 where
  TransformerLM
    :: forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
     . { tEmbedding    :: Embedding ('Just paddingIdx) numEmbeds embedDim 'Learned dtype device
       , tPosEmbedding :: Embedding 'Nothing           2048      embedDim 'Constant dtype device
       , tDropout      :: Dropout
       , tLayers       :: HList (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
       , tProj         :: Linear embedDim seqLen dtype device
       }
    -> TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
 deriving (Generic)

instance
  ( paddingIdx <= numEmbeds
  , 1 <= numEmbeds - paddingIdx
  , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
  , (Div embedDim 2 * 2) ~ embedDim
  , All KnownNat '[ffnDim, paddingIdx, numEmbeds, embedDim, seqLen]
  , HReplicate numAttnLayers (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)
  , A.Randomizable (HList (HReplicateR numAttnLayers (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)))
                   (HList (HReplicateR numAttnLayers (TransformerLMLayer     embedDim numHeads ffnDim dtype device)))
  , KnownDType dtype
  , RandDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device 'D.Float
  , BasicArithmeticDTypeIsValid device 'D.Float
  , KnownDevice device
  ) => A.Randomizable (TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device)
                      (TransformerLM     numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device)
 where
  sample TransformerLMSpec {..} =
    TransformerLM
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @( 'Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @ 'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample lmDropoutSpec
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample LinearSpec

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
          . cmul (- log (10000 :: Double) / (fromInteger . natVal $ Proxy @(Div embedDim 2)))
          . linspace @(Div embedDim 2) (0 :: Int)
          $ natValI @((Div embedDim 2) - 1)
      radians = mul positions scalingFactors
      weights = stack @2 (sin radians :. cos radians :. HNil)
  in  reshape weights

logits
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
         FoldLayers
         (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDevice device
     )
  => TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
  -> Bool
  -> Tensor device 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor device dtype '[batchSize, seqLen, seqLen])
logits TransformerLM {..} train input = do
  hidden <-
    transpose @0 @1
      <$> getHidden @numAttnLayers @numHeads @ffnDim
            tEmbedding
            tPosEmbedding
            tDropout
            train
            tLayers
            input
  return $ linear tProj hidden
