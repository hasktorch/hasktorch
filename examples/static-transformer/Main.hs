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

module Main where

import           Prelude                 hiding ( (.), id )
import           Control.Arrow
import           Control.Category
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import           Torch.Static
import           Torch.Static.Native     hiding ( linear )
import           Torch.Static.Factories
import           Torch.Static.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D


--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data Activation (dtype :: D.DType)
 where
  Activation
    :: forall dtype
     . { unActivation :: forall shape . Tensor dtype shape -> Tensor dtype shape }
    -> Activation dtype

instance Show (Activation dtype) where
  show _ = mempty

instance A.Parameterized (Activation dtype) where
  flattenParameters _ = []
  replaceOwnParameters = return

data MultiheadAttentionSpec (dtype :: D.DType)
                            (embedDim :: Nat) (numHeads :: Nat)
 where
  MultiheadAttentionSpec
    :: { mhDropoutProbSpec :: Double }
    -> MultiheadAttentionSpec dtype embedDim numHeads
 deriving (Show, Eq)

data MultiheadAttention (dtype :: D.DType)
                        (embedDim :: Nat) (numHeads :: Nat)
 where
  MultiheadAttention
    :: { mhInProj :: Linear dtype embedDim (embedDim * 3)
       , mhOutProj :: Linear dtype embedDim embedDim
       , mhDropout :: Dropout
       }
    -> MultiheadAttention dtype embedDim numHeads
 deriving (Show, Generic)

multiheadAttention
  :: forall dtype embedDim numHeads seqLen batchSize headDim
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , KnownDType dtype
     , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
     )
  => MultiheadAttention dtype embedDim numHeads
  -> Bool
  -> Tensor 'D.Bool '[seqLen, batchSize]
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim], Tensor dtype '[batchSize, seqLen, seqLen])
multiheadAttention MultiheadAttention {..} train keyPaddingMask input = do
  let q :. k :. v :. HNil = chunk @3 @2 . linear mhInProj $ input
  attnWeights <-
    Torch.Static.NN.dropout mhDropout train
      . softmax @2
      . maskKeyPaddings
      . maskFutureTimesteps
      $ (mul scaling $ f q) `matmul` (transpose @1 @2 $ f k)
  let attn =
        linear mhOutProj
          . reshape @'[seqLen, batchSize, embedDim]
          . transpose @0 @1
          $ attnWeights `matmul` (f v)
      avgAttnWeights =
        mul (pow (-1 :: Double) (fromInteger . natVal $ Proxy @numHeads) :: Tensor dtype '[])
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
  scaling :: Tensor dtype '[]
  scaling = pow (-1 / 2 :: Double) (fromInteger . natVal $ Proxy @headDim)
  futureTimestampMask =
    toDType @D.Bool . triu 1 $ ones @D.Int8 @'[seqLen, seqLen]

multiheadAttention'
  :: forall dtype embedDim numHeads seqLen batchSize headDim
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , KnownDType dtype
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     )
  => MultiheadAttention dtype embedDim numHeads
  -> Bool
  -> Kleisli
       IO
       ( Tensor 'D.Bool '[seqLen, batchSize]
       , Tensor dtype '[seqLen, batchSize, embedDim]
       )
       ( Tensor dtype '[seqLen, batchSize, embedDim]
       , Tensor dtype '[batchSize, seqLen, seqLen]
       )
multiheadAttention' MultiheadAttention {..} train =
  second (linear mhInProj ^>> split)
    >>> assoc
    >>> first attnWeights
    >>> (attn &&& (first avgAttnWeights >>^ fst))
 where
  split = arr $ \t ->
    let q :. k :. v :. HNil = chunk @3 @2 t
    in  ((q, k), v)
  assoc = arr $ \(keyPaddingMask, ((q, k), v)) -> ((keyPaddingMask, (q, k)), v)
  attnWeights =
    second dotProduct
      >>> second maskFutureTimestamps
      >>> maskKeyPaddings
      >>> softmax @2
      ^>> Kleisli (Torch.Static.NN.dropout mhDropout train)
   where
    dotProduct =
      let scaling = pow (-1 / 2 :: Double) (fromInteger . natVal $ Proxy @headDim) :: Tensor dtype '[]
      in  ((f >>^ mul scaling) *** (f >>^ transpose @1 @2)) >>^ uncurry matmul
    maskFutureTimestamps =
      let futureTimestampMask = toDType @D.Bool . triu 1 $ ones @D.Int8 @'[seqLen, seqLen]
      in  arr $ maskedFill futureTimestampMask (-1 / 0 :: Double)
    maskKeyPaddings =
      first (transpose @0 @1 ^>> unsqueeze @1 ^>> unsqueeze @2 ^>> returnA)
        >>> second (arr $ reshape @'[batchSize, numHeads, seqLen, seqLen])
        >>^ (uncurry $ flip maskedFill (-1 / 0 :: Double))
        >>^ reshape @'[batchSize * numHeads, seqLen, seqLen]
  attn =
    (id *** f)
      >>^ uncurry matmul
      >>^ transpose @0 @1
      >>^ reshape @'[seqLen, batchSize, embedDim]
      >>^ linear mhOutProj
  avgAttnWeights =
    let factor = pow (-1 :: Double) (fromInteger . natVal $ Proxy @numHeads) :: Tensor dtype '[]
    in  reshape @'[batchSize, numHeads, seqLen, seqLen]
          ^>> sumDim @1
          ^>> mul factor
          ^>> returnA
  f = reshape @'[seqLen, batchSize * numHeads, headDim] ^>> transpose @0 @1 ^>> returnA

instance A.Parameterized (MultiheadAttention dtype embedDim numHeads)  

instance ( KnownDType dtype
         , KnownNat embedDim
         , KnownNat numHeads
         )
  => A.Randomizable (MultiheadAttentionSpec dtype embedDim numHeads)
                    (MultiheadAttention     dtype embedDim numHeads)
 where
  sample MultiheadAttentionSpec {..} =
    MultiheadAttention
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (DropoutSpec mhDropoutProbSpec)

data TransformerLMLayerSpec (dtype :: D.DType)
                            (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
 where
  TransformerLMLayerSpec
    :: { tMHDropoutProbSpec :: Double
       , tAttnDropoutProbSpec :: Double
       , tLNEpsSpec :: Double
       , tMLPDropout0ProbSpec :: Double
       , tMLPDropout1ProbSpec :: Double
       , tMLPActivation0Spec :: Activation dtype
       , tMLPActivation1Spec :: Activation dtype
       }
    -> TransformerLMLayerSpec dtype embedDim numHeads ffnDim
 deriving (Show)

data TransformerLMLayer (dtype :: D.DType)
                        (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
 where
  TransformerLMLayer
    :: { tAttn :: MultiheadAttention dtype embedDim numHeads
       , tAttnDropout :: Dropout
       , tLN0 :: LayerNorm dtype '[embedDim]
       , tLN1 :: LayerNorm dtype '[embedDim]
       , tMLP :: TransformerLMMLP dtype embedDim ffnDim
       }
    -> TransformerLMLayer dtype embedDim numHeads ffnDim
 deriving (Show, Generic)

transformerLMLayer
  :: forall dtype numHeads ffnDim embedDim headDim seqLen batchSize
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , KnownDType dtype
     , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
     , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
     )
  => TransformerLMLayer dtype embedDim numHeads ffnDim
  -> Bool
  -> Tensor 'D.Bool '[seqLen, batchSize]
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMLayer TransformerLMLayer {..} train keyPaddingMask input = do
  (attn, _) <- multiheadAttention tAttn train keyPaddingMask input
  x         <- Torch.Static.NN.dropout tAttnDropout train attn
  let x' = Torch.Static.NN.layerNorm tLN0 (x `add` input)
  x''       <- transformerLMMLP tMLP train x'
  return $ Torch.Static.NN.layerNorm tLN1 (x'' `add` x')

transformerLMLayer'
  :: forall dtype numHeads ffnDim embedDim headDim seqLen batchSize
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , KnownDType dtype
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
     )
  => TransformerLMLayer dtype embedDim numHeads ffnDim
  -> Bool
  -> Kleisli
       IO
       ( Tensor 'D.Bool '[seqLen, batchSize]
       , Tensor dtype '[seqLen, batchSize, embedDim]
       )
       (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMLayer' TransformerLMLayer {..} train =
  (arr snd &&& attn)
    >>> uncurry add
    ^>> Torch.Static.NN.layerNorm tLN0
    ^>> (id &&& transformerLMMLP' tMLP train)
    >>^ uncurry add
    >>^ Torch.Static.NN.layerNorm tLN1
 where
  attn = multiheadAttention' tAttn train >>> arr fst >>> Kleisli
    (Torch.Static.NN.dropout tAttnDropout train)

instance A.Parameterized (TransformerLMLayer dtype embedDim numHeads ffnDim)

instance ( KnownDType dtype
         , KnownNat embedDim
         , KnownNat numHeads
         , KnownNat ffnDim
         )
  => A.Randomizable (TransformerLMLayerSpec dtype embedDim numHeads ffnDim)
                    (TransformerLMLayer     dtype embedDim numHeads ffnDim)
 where
  sample TransformerLMLayerSpec {..} =
    let mhDropoutProbSpec = tMHDropoutProbSpec
        dropoutProbSpec = tAttnDropoutProbSpec
        layerNormEpsSpec = tLNEpsSpec
        tDropout0ProbSpec = tMLPDropout0ProbSpec
        tDropout1ProbSpec = tMLPDropout1ProbSpec
        tActivation0Spec = tMLPActivation0Spec
        tActivation1Spec = tMLPActivation1Spec
    in  TransformerLMLayer
          <$> A.sample MultiheadAttentionSpec {..}
          <*> A.sample DropoutSpec {..}
          <*> A.sample LayerNormSpec {..}
          <*> A.sample LayerNormSpec {..}
          <*> A.sample TransformerLMMLPSpec {..}

data TransformerLMMLPSpec (dtype :: D.DType)
                          (embedDim :: Nat) (ffnDim :: Nat)
 where
  TransformerLMMLPSpec
    :: forall dtype embedDim ffnDim
     . { tDropout0ProbSpec :: Double
       , tDropout1ProbSpec :: Double
       , tActivation0Spec :: Activation dtype
       , tActivation1Spec :: Activation dtype
       }
    -> TransformerLMMLPSpec dtype embedDim ffnDim

data TransformerLMMLP (dtype :: D.DType)
                      (embedDim :: Nat) (ffnDim :: Nat)
 where
  TransformerLMMLP
    :: forall dtype embedDim ffnDim
     . { tLinear0 :: Linear dtype embedDim ffnDim
       , tLinear1 :: Linear dtype ffnDim embedDim
       , tDropout0 :: Dropout
       , tDropout1 :: Dropout
       , tActivation0 :: Activation dtype
       , tActivation1 :: Activation dtype
       }
    -> TransformerLMMLP dtype embedDim ffnDim
 deriving (Show, Generic)

transformerLMMLP
  :: forall dtype embedDim ffnDim seqLen batchSize
   . TransformerLMMLP dtype embedDim ffnDim
  -> Bool
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMMLP TransformerLMMLP {..} train input =
  Torch.Static.NN.dropout tDropout1 train
    .   unActivation tActivation1
    .   linear tLinear1
    =<< Torch.Static.NN.dropout tDropout0 train
    .   unActivation tActivation0
    .   linear tLinear0
    =<< pure input

transformerLMMLP'
  :: forall dtype embedDim ffnDim seqLen batchSize
   . TransformerLMMLP dtype embedDim ffnDim
  -> Bool
  -> Kleisli
       IO
       (Tensor dtype '[seqLen, batchSize, embedDim])
       (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMMLP' TransformerLMMLP {..} train =
  linear tLinear0
    ^>> unActivation tActivation0
    ^>> Kleisli (Torch.Static.NN.dropout tDropout0 train)
    >>> linear tLinear1
    ^>> unActivation tActivation1
    ^>> Kleisli (Torch.Static.NN.dropout tDropout1 train)

instance A.Parameterized (TransformerLMMLP dtype embedDim ffnDim)

instance ( KnownDType dtype
         , KnownNat embedDim
         , KnownNat ffnDim
         )
  => A.Randomizable (TransformerLMMLPSpec dtype embedDim ffnDim)
                    (TransformerLMMLP     dtype embedDim ffnDim)
 where
  sample TransformerLMMLPSpec {..} =
    TransformerLMMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (DropoutSpec tDropout0ProbSpec)
      <*> A.sample (DropoutSpec tDropout1ProbSpec)
      <*> pure tActivation0Spec
      <*> pure tActivation1Spec

data FoldLayers = FoldLayers { foldLayersTrain :: Bool }

instance ( 1 <= numHeads
         , embedDim ~ (headDim * numHeads)
         , Mod (embedDim * 3) 3 ~ 0
         , Div (embedDim * 3) 3 ~ embedDim
         , KnownDType dtype
         , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
         , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
         )
    => Apply
         FoldLayers
         (TransformerLMLayer dtype embedDim numHeads ffnDim)
         ((Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim]) -> IO (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim]))
 where
  apply FoldLayers {..} layer = \(keyPaddingMask, input) -> do
    output <- transformerLMLayer layer foldLayersTrain keyPaddingMask input
    return (keyPaddingMask, output)

data FoldLayers' = FoldLayers' { foldLayersTrain' :: Bool }

instance ( 1 <= numHeads
         , embedDim ~ (headDim * numHeads)
         , Mod (embedDim * 3) 3 ~ 0
         , Div (embedDim * 3) 3 ~ embedDim
         , KnownDType dtype
         , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
         , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
         )
    => Apply
         FoldLayers'
         (TransformerLMLayer dtype embedDim numHeads ffnDim)
         (Kleisli IO (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim]) (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim]))
 where
  apply FoldLayers' {..} layer = arr fst &&& transformerLMLayer' layer foldLayersTrain'

getHidden
  :: forall
       numAttnLayers
       numHeads
       ffnDim
       paddingIdx
       dtype
       numEmbeds
       embedDim
       seqLen
       batchSize
   . ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
     , paddingIdx + 1 <= numEmbeds
     , 1 <= seqLen
     , HFoldrM
         IO
         FoldLayers
         (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
     )
  => Embedding ('Just paddingIdx) dtype numEmbeds embedDim
  -> Embedding 'Nothing dtype 2048 embedDim
  -> Dropout
  -> Bool
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
  -> Tensor 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
getHidden embedding posEmbedding dropout train layers input = do
  let srcTokens = transpose @0 @1 input
      src       = embed embedding srcTokens
      positions = expand @'[seqLen, batchSize, embedDim] True
                    . unsqueeze @1
                    . embed posEmbedding
                    . toDType @D.Int64
                    . linspace @seqLen 0
                    . fromIntegral
                    $ natValI @(seqLen - 1)
  x <- Torch.Static.NN.dropout dropout train (src `add` positions)
  let keyPaddingMask = srcTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor 'D.Int64 '[])
  (_, x') <- hfoldrM (FoldLayers train) (keyPaddingMask, x) layers
  return x'

getHidden'
  :: forall
       numAttnLayers
       numHeads
       ffnDim
       paddingIdx
       dtype
       numEmbeds
       embedDim
       seqLen
       batchSize
   . ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
     , paddingIdx + 1 <= numEmbeds
     , 1 <= seqLen
     , HFoldrM'
         IO
         FoldLayers'
         (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
     )
  => Embedding ('Just paddingIdx) dtype numEmbeds embedDim
  -> Embedding 'Nothing dtype 2048 embedDim
  -> Dropout
  -> Bool
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
  -> Kleisli IO (Tensor 'D.Int64 '[batchSize, seqLen]) (Tensor dtype '[seqLen, batchSize, embedDim])
getHidden' embedding posEmbedding dropout train layers =
  transpose @0 @1
    ^>> (mkKeyPaddingMask &&& mkInput)
    >>> hfoldrM' (FoldLayers' train) layers
    >>^ snd
 where
  mkKeyPaddingMask =
    arr (==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor 'D.Int64 '[]))
  mkInput = embed embedding ^>> add positions ^>> Kleisli (Torch.Static.NN.dropout dropout train)
  positions =
    expand @'[seqLen, batchSize, embedDim] True
      . unsqueeze @1
      . embed posEmbedding
      . toDType @D.Int64
      . linspace @seqLen 0
      . fromIntegral
      $ natValI @(seqLen - 1)

data TransformerLMSpec
       (numAttnLayers :: Nat)
       (numHeads :: Nat)
       (ffnDim :: Nat)
       (paddingIdx :: Nat)
       (dtype :: D.DType)
       (numEmbeds :: Nat)
       (embedDim :: Nat)
       (seqLen :: Nat)
 where
  TransformerLMSpec
    :: forall
         numAttnLayers
         numHeads
         ffnDim
         paddingIdx
         dtype
         numEmbeds
         embedDim
         seqLen
     . { tDropoutProbSpec :: Double
       , tLayerSpec :: TransformerLMLayerSpec dtype embedDim numHeads ffnDim
       }
    -> TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx dtype numEmbeds embedDim seqLen
 deriving (Show)

data TransformerLM
       (numAttnLayers :: Nat)
       (numHeads :: Nat)
       (ffnDim :: Nat)
       (paddingIdx :: Nat)
       (dtype :: D.DType)
       (numEmbeds :: Nat)
       (embedDim :: Nat)
       (seqLen :: Nat)
 where
  TransformerLM
    :: forall
         numAttnLayers
         numHeads
         ffnDim
         paddingIdx
         dtype
         numEmbeds
         embedDim
         seqLen
     . { tEmbedding :: Embedding ('Just paddingIdx) dtype numEmbeds embedDim
       , tPosEmbedding :: Embedding 'Nothing dtype 2048 embedDim
       , tDropout :: Dropout
       , tLayers :: HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
       , tProj :: Linear dtype embedDim seqLen
       }
    -> TransformerLM numAttnLayers numHeads ffnDim paddingIdx dtype numEmbeds embedDim seqLen
 deriving (Generic)

instance (A.Parameterized (HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))))
  => A.Parameterized (TransformerLM numAttnLayers
                                    numHeads
                                    ffnDim
                                    paddingIdx
                                    dtype
                                    numEmbeds
                                    embedDim
                                    seqLen)

instance ( paddingIdx <= numEmbeds
         , 1 <= numEmbeds - paddingIdx
         , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
         , KnownNat ffnDim
         , KnownNat paddingIdx
         , KnownDType dtype
         , KnownNat numEmbeds
         , KnownNat embedDim
         , KnownNat seqLen
         , HReplicate'
             numAttnLayers
             (TransformerLMLayerSpec dtype embedDim numHeads ffnDim)
             (HReplicateR
               numAttnLayers
               (TransformerLMLayerSpec dtype embedDim numHeads ffnDim))
         , A.Randomizable
             (HList
               (HReplicateR
                 numAttnLayers
                 (TransformerLMLayerSpec dtype embedDim numHeads ffnDim)))
             (HList
               (HReplicateR
                 numAttnLayers
                 (TransformerLMLayer dtype embedDim numHeads ffnDim)))
         )
  => A.Randomizable (TransformerLMSpec numAttnLayers
                                       numHeads
                                       ffnDim
                                       paddingIdx
                                       dtype
                                       numEmbeds
                                       embedDim
                                       seqLen)
                    (TransformerLM     numAttnLayers
                                       numHeads
                                       ffnDim
                                       paddingIdx
                                       dtype
                                       numEmbeds
                                       embedDim
                                       seqLen)
 where
  sample TransformerLMSpec {..} =
    TransformerLM
      <$> A.sample (EmbeddingSpec @( 'Just paddingIdx))
      <*> A.sample (EmbeddingSpec @ 'Nothing)
      <*> A.sample (DropoutSpec tDropoutProbSpec)
      <*> A.sample
            (hReplicate (Proxy @numAttnLayers) tLayerSpec :: HList
                ( HReplicateR
                    numAttnLayers
                    (TransformerLMLayerSpec dtype embedDim numHeads ffnDim)
                )
            )
      <*> A.sample LinearSpec

logits
  :: forall
       numAttnLayers
       numHeads
       ffnDim
       paddingIdx
       dtype
       numEmbeds
       embedDim
       seqLen
       batchSize
   . ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
     , paddingIdx + 1 <= numEmbeds
     , 1 <= seqLen
     , HFoldrM
         IO
         FoldLayers
         (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
     )
  => TransformerLM numAttnLayers numHeads ffnDim paddingIdx dtype numEmbeds embedDim seqLen
  -> Bool
  -> Tensor 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor dtype '[batchSize, seqLen, seqLen])
logits TransformerLM {..} train input = do
  hidden <-
    transpose @0 @1
      <$> getHidden @numAttnLayers @numHeads @ffnDim -- TODO: these type applications shouldn't be necessary
            tEmbedding
            tPosEmbedding
            tDropout
            train
            tLayers
            input
  return $ linear tProj hidden

logits'
  :: forall
       numAttnLayers
       numHeads
       ffnDim
       paddingIdx
       dtype
       numEmbeds
       embedDim
       seqLen
       batchSize
   . ( All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
     , paddingIdx + 1 <= numEmbeds
     , 1 <= seqLen
     , HFoldrM'
         IO
         FoldLayers'
         (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
     )
  => TransformerLM numAttnLayers numHeads ffnDim paddingIdx dtype numEmbeds embedDim seqLen
  -> Bool
  -> Kleisli IO (Tensor 'D.Int64 '[batchSize, seqLen]) (Tensor dtype '[batchSize, seqLen, seqLen])
logits' TransformerLM {..} train =
  getHidden' @numAttnLayers @numHeads @ffnDim tEmbedding
                                              tPosEmbedding
                                              tDropout
                                              train
                                              tLayers
    >>^ transpose @0 @1
    >>^ linear tProj

toBackend
  :: forall t . (ATen.Castable t (ForeignPtr ATen.Tensor)) => String -> t -> t
toBackend backend t = unsafePerformIO $ case backend of
  "CUDA" -> ATen.cast1 ATen.tensor_cuda t
  _      -> ATen.cast1 ATen.tensor_cpu t

crossEntropyLoss
  :: forall paddingIdx batchSize seqLen dtype
   . (KnownNat paddingIdx, KnownNat batchSize, KnownNat seqLen, KnownDType dtype)
  => String
  -> Tensor dtype '[batchSize, seqLen, seqLen]
  -> Tensor 'D.Int64 '[batchSize, seqLen]
  -> Tensor dtype '[]
crossEntropyLoss backend result target =
  nll_loss @D.ReduceMean @dtype @batchSize @seqLen @'[seqLen]
    (logSoftmax @1 result)
    target
    (toBackend backend ones)
    (natValI @paddingIdx)

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

type NumAttnLayers = 1
type NumHeads = 1
type FFNDim = 1
type PaddingIdx = 0
type NumEmbeds = 10
type EmbedDim = 5
type SeqLen = 1

type Model
  = TransformerLM
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      'D.Float
      NumEmbeds
      EmbedDim
      SeqLen

type ModelSpec
  = TransformerLMSpec
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      'D.Float
      NumEmbeds
      EmbedDim
      SeqLen

data Data

type BatchSize = 1

main = do
  backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
  let backend = case backend' of
        Right "CUDA" -> "CUDA"
        _            -> "CPU"
      numIters = 1
  init  <- A.sample spec :: IO Model
  init' <- A.replaceParameters init <$> traverse
    (A.makeIndependent . toBackend backend . A.toDependent)
    (A.flattenParameters init)
  (_trained, _) <- foldLoop (init', undefined) numIters $ \(state, _) i -> do
    trainingLoss <- computeLoss @BatchSize backend state True undefined undefined
    let flat_parameters = A.flattenParameters state
    let gradients       = A.grad (toDynamic trainingLoss) flat_parameters
    new_flat_parameters <- mapM A.makeIndependent
      $ A.sgd 1e-01 flat_parameters gradients
    return (A.replaceParameters state new_flat_parameters, undefined)
  return ()
 where
  spec :: ModelSpec
  spec = TransformerLMSpec
    { tDropoutProbSpec = 0.0
    , tLayerSpec       = TransformerLMLayerSpec
                           { tMHDropoutProbSpec   = 0.0
                           , tAttnDropoutProbSpec = 0.0
                           , tLNEpsSpec           = 0.0
                           , tMLPDropout0ProbSpec = 0.0
                           , tMLPDropout1ProbSpec = 0.0
                           , tMLPActivation0Spec  = Activation relu
                           , tMLPActivation1Spec  = Activation relu
                           }
    }
  computeLoss
    :: forall batchSize
     . (KnownNat batchSize, EndsWith '[batchSize, EmbedDim] '[EmbedDim])
    => String
    -> Model
    -> Bool
    -> [Int]
    -> Data
    -> IO (Tensor 'D.Float '[])
  computeLoss backend state train _indexes _data = do
    let input  = toBackend backend (undefined :: Tensor 'D.Int64 '[batchSize, SeqLen])
        target = toBackend backend (undefined :: Tensor 'D.Int64 '[batchSize, SeqLen])
    result <- logits state train input
    return $ crossEntropyLoss @PaddingIdx backend result target
