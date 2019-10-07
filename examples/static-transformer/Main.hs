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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module Main where

import           Prelude                 hiding ( (.), id, tanh )
import           Control.Arrow
import           Control.Category
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.List                      ( foldl'
                                                , scanl'
                                                , intersperse
                                                )
import           Data.Reflection
import           Data.Proxy
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra

import           Torch.Static
import           Torch.Static.Native     hiding ( linear )
import           Torch.Static.Factories
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D


--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------


newtype Parameter dtype shape = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter dtype shape -> Tensor dtype shape
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t

instance A.Parameterized (Parameter dtype shape) where
  flattenParameters (Parameter x) = [x]
  replaceOwnParameters _ = Parameter <$> A.nextParameter

data MultiheadAttention (dtype :: D.DType) (embedDim :: Nat) (numHeads :: Nat) where
  MultiheadAttention
    :: (1 <= numHeads)
    => { mhInProj :: Linear dtype embedDim (embedDim * 3)
       , mhOutProj :: Linear dtype embedDim embedDim
       , mhDropout :: Dropout
       }
    -> MultiheadAttention dtype embedDim numHeads

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
  -> Tensor 'D.Bool '[seqLen, batchSize]
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim], Tensor dtype '[batchSize, seqLen, seqLen])
multiheadAttention MultiheadAttention {..} keyPaddingMask input = do
  let q :. k :. v :. HNil = chunk @3 @2 . linear mhInProj $ input
  attnWeights <-
    Main.dropout mhDropout
      . softmax @2
      . maskKeyPaddings
      . maskFutureTimestamps
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
  maskFutureTimestamps = maskedFill futureTimestampMask (-1 / 0 :: Double)
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
  -> Kleisli
       IO
       ( Tensor 'D.Bool '[seqLen, batchSize]
       , Tensor dtype '[seqLen, batchSize, embedDim]
       )
       ( Tensor dtype '[seqLen, batchSize, embedDim]
       , Tensor dtype '[batchSize, seqLen, seqLen]
       )
multiheadAttention' MultiheadAttention {..} =
  Kleisli $ \(keyPaddingMask, input) -> do
    let q :. k :. v :. HNil = chunk @3 @2 . linear mhInProj $ input
        maskKeyPaddings =
          reshape @'[batchSize * numHeads, seqLen, seqLen]
            . maskedFill
                (unsqueeze @2 . unsqueeze @1 . transpose @0 @1 $ keyPaddingMask)
                (-1 / 0 :: Double)
            . reshape @'[batchSize, numHeads, seqLen, seqLen]
    attnWeights <-
      Main.dropout mhDropout
      .        softmax @2
      .        maskKeyPaddings
      .        maskFutureTimestamps
      $        (mul scaling $ f q)
      `matmul` (transpose @1 @2 $ f k)
    let attn =
          linear mhOutProj
            .        reshape @'[seqLen, batchSize, embedDim]
            .        transpose @0 @1
            $        attnWeights
            `matmul` (f v)
        avgAttnWeights =
          mul
              (pow (-1 :: Double) (fromInteger . natVal $ Proxy @numHeads) :: Tensor
                  dtype
                  '[]
              )
            . sumDim @1
            . reshape @'[batchSize, numHeads, seqLen, seqLen]
            $ attnWeights
    return (attn, avgAttnWeights)
 where
  chunk' :: Kleisli IO (Tensor dtype '[seqLen, batchSize, embedDim]) ((Tensor dtype '[seqLen, batchSize, embedDim], Tensor dtype '[seqLen, batchSize, embedDim]), Tensor dtype '[seqLen, batchSize, embedDim])
  chunk' = arr $ \input -> let q :. k :. v :. HNil = (chunk @3 @2 . linear mhInProj) input in ((q, k), v)

  mkAttnWeights
    :: Kleisli
         IO
         ( Tensor dtype '[seqLen, batchSize, embedDim]
         , Tensor dtype '[seqLen, batchSize, embedDim]
         )
         (Tensor dtype '[batchSize * numHeads, seqLen, seqLen])
  mkAttnWeights = 
    (arr (mul scaling . f) *** arr (transpose @1 @2 . f))
      >>> uncurry matmul
      ^>> maskFutureTimestamps
      ^>> softmax @2
      ^>> Kleisli (Main.dropout mhDropout)
  -- mkAttenWeights'
  --   :: Kleisli
  --        IO
  --        ( Tensor 'D.Bool '[seqLen, batchSize]
  --        , ( Tensor dtype '[seqLen, batchSize, embedDim]
  --          , Tensor dtype '[seqLen, batchSize, embedDim]
  --          )
  --        )
  --        (Tensor dtype '[batchSize * numHeads, seqLen, seqLen])
  -- mkAttenWeights' = 
  --     ^>> softmax @2
  --     ^>> Kleisli (Main.dropout mhDropout)

  maskedFill' :: Kleisli IO (Tensor 'D.Bool '[seqLen, batchSize], Tensor dtype '[batchSize, numHeads, seqLen, seqLen]) (Tensor dtype '[batchSize, numHeads, seqLen, seqLen])
  maskedFill' = arr $ \(keyPaddingMask, input) -> maskedFill (unsqueeze @2 . unsqueeze @1 . transpose @0 @1 $ keyPaddingMask) (-1 / 0 :: Double) input

  maskFutureTimestamps = maskedFill futureTimestampMask (-1 / 0 :: Double)
  f = transpose @0 @1 . reshape @'[seqLen, batchSize * numHeads, headDim]
  scaling :: Tensor dtype '[]
  scaling = pow (-1 / 2 :: Double) (fromInteger . natVal $ Proxy @headDim)
  futureTimestampMask =
    toDType @D.Bool . triu 1 $ ones @D.Int8 @'[seqLen, seqLen]

data Dropout where
  Dropout
    :: { dropoutProb :: Double
       , dropoutTrain :: Bool
       }
    -> Dropout

dropout :: Dropout -> Tensor dtype shape -> IO (Tensor dtype shape)
dropout Dropout {..} = Torch.Static.Native.dropout dropoutProb dropoutTrain

data TransformerLMLayer (dtype :: D.DType) (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat) where
  TransformerLMLayer
    :: { tAttn :: MultiheadAttention dtype embedDim numHeads
       , tAttnDropout :: Dropout
       , tLN0 :: LayerNorm dtype '[embedDim]
       , tLN1 :: LayerNorm dtype '[embedDim]
       , tMLP :: TransformerLMMLP dtype embedDim ffnDim
       }
    -> TransformerLMLayer dtype embedDim numHeads ffnDim

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
  -> Tensor 'D.Bool '[seqLen, batchSize]
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMLayer TransformerLMLayer {..} keyPaddingMask input = do
  (attn, _) <- multiheadAttention tAttn keyPaddingMask input
  x         <- Main.dropout tAttnDropout attn
  let x' = Main.layerNorm tLN0 (x `add` input)
  x''       <- transformerLMMLP tMLP x'
  return $ Main.layerNorm tLN1 (x'' `add` x')

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
  -> Kleisli
       IO
       ( Tensor 'D.Bool '[seqLen, batchSize]
       , Tensor dtype '[seqLen, batchSize, embedDim]
       )
       (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMLayer' TransformerLMLayer {..} =
  (arr snd &&& attn)
    >>> uncurry add
    ^>> Main.layerNorm tLN0
    ^>> (id &&& transformerLMMLP' tMLP)
    >>^ uncurry add
    >>^ Main.layerNorm tLN1
 where
  attn = multiheadAttention' tAttn >>> arr fst >>> Kleisli (Main.dropout tAttnDropout)

data TransformerLMMLP (dtype :: D.DType) (embedDim :: Nat) (ffnDim :: Nat) where
  TransformerLMMLP
    :: { tLinear0 :: Linear dtype embedDim ffnDim
       , tLinear1 :: Linear dtype ffnDim embedDim
       , tDropout0 :: Dropout
       , tDropout1 :: Dropout
       , tActivation0 :: forall shape . Tensor dtype shape -> Tensor dtype shape
       , tActivation1 :: forall shape . Tensor dtype shape -> Tensor dtype shape
       }
    -> TransformerLMMLP dtype embedDim ffnDim

transformerLMMLP
  :: forall dtype embedDim ffnDim seqLen batchSize
   . TransformerLMMLP dtype embedDim ffnDim
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMMLP TransformerLMMLP {..} input =
  Main.dropout tDropout1
    .   tActivation1
    .   linear tLinear1
    =<< Main.dropout tDropout0
    .   tActivation0
    .   linear tLinear0
    =<< pure input

transformerLMMLP'
  :: forall dtype embedDim ffnDim seqLen batchSize
   . TransformerLMMLP dtype embedDim ffnDim
  -> Kleisli
       IO
       (Tensor dtype '[seqLen, batchSize, embedDim])
       (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMMLP' TransformerLMMLP {..} =
  linear tLinear0
    ^>> tActivation0
    ^>> Kleisli (Main.dropout tDropout0)
    >>> linear tLinear1
    ^>> tActivation1
    ^>> Kleisli (Main.dropout tDropout1)

data Embedding (paddingIdx :: Maybe Nat) (dtype :: D.DType) (numEmbeds :: Nat) (embedDim :: Nat) where
  Embedding
    :: forall paddingIdx dtype numEmbeds embedDim
     . (PaddingIdxCheck paddingIdx numEmbeds)
    => { embedWeights :: Parameter dtype '[numEmbeds, embedDim] }
    -> Embedding paddingIdx dtype numEmbeds embedDim

embed
  :: forall paddingIdx dtype shape numEmbeds embedDim
   . ( KnownMaybeNat paddingIdx
     , PaddingIdxCheck paddingIdx numEmbeds
     )
  => Embedding paddingIdx dtype numEmbeds embedDim
  -> Tensor 'D.Int64 shape
  -> Tensor dtype (Reverse (embedDim ': (Reverse shape)))
embed Embedding {..} input = embedding @paddingIdx
  False
  False
  (toDependent embedWeights)
  input

data FoldLayers = FoldLayers

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
  apply _ layer = \(keyPaddingMask, input) -> do
    output <- transformerLMLayer layer keyPaddingMask input
    return (keyPaddingMask, output)

data FoldLayers' = FoldLayers'

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
  apply _ layer = arr fst &&& transformerLMLayer' layer

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
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
  -> Tensor 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
getHidden embedding posEmbedding dropout layers input = do
  let srcTokens = transpose @0 @1 input
      src       = embed embedding srcTokens
      positions = expand @'[seqLen, batchSize, embedDim] True
                    . unsqueeze @1
                    . embed posEmbedding
                    . toDType @D.Int64
                    . linspace @seqLen 0
                    . fromIntegral
                    $ natValI @(seqLen - 1)
  x <- Main.dropout dropout (src `add` positions)
  let keyPaddingMask = srcTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor 'D.Int64 '[])
  (_, x') <- hfoldrM FoldLayers (keyPaddingMask, x) layers
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
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer dtype embedDim numHeads ffnDim))
  -> Kleisli IO (Tensor 'D.Int64 '[batchSize, seqLen]) (Tensor dtype '[seqLen, batchSize, embedDim])
getHidden' embedding posEmbedding dropout layers =
  transpose @0 @1
    ^>> (mkKeyPaddingMask &&& mkInput)
    >>> hfoldrM' FoldLayers' layers
    >>^ snd
 where
  mkKeyPaddingMask =
    arr (==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor 'D.Int64 '[]))
  mkInput = embed embedding ^>> add positions ^>> Kleisli (Main.dropout dropout)
  positions =
    expand @'[seqLen, batchSize, embedDim] True
      . unsqueeze @1
      . embed posEmbedding
      . toDType @D.Int64
      . linspace @seqLen 0
      . fromIntegral
      $ natValI @(seqLen - 1)

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
  -> Tensor 'D.Int64 '[batchSize, seqLen]
  -> IO (Tensor dtype '[batchSize, seqLen, seqLen])
logits TransformerLM {..} input = do
  hidden <-
    transpose @0 @1
      <$> getHidden @numAttnLayers @numHeads @ffnDim -- TODO: these type applications shouldn't be necessary
            tEmbedding
            tPosEmbedding
            tDropout
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
  -> Kleisli IO (Tensor 'D.Int64 '[batchSize, seqLen]) (Tensor dtype '[batchSize, seqLen, seqLen])
logits' TransformerLM {..} =
  getHidden' @numAttnLayers @numHeads @ffnDim tEmbedding
                                              tPosEmbedding
                                              tDropout
                                              tLayers
    >>^ transpose @0 @1
    >>^ linear tProj

data LayerNorm (dtype :: D.DType) (normalizedShape :: [Nat]) where
  LayerNorm
    :: { layerNormWeight :: Parameter dtype normalizedShape
       , layerNormBias :: Parameter dtype normalizedShape
       , layerNormEps :: Double
       }
    -> LayerNorm dtype normalizedShape

layerNorm
  :: forall normalizedShape dtype shape
   . ( EndsWith shape normalizedShape
     , KnownShape normalizedShape
     )
  => LayerNorm dtype normalizedShape
  -> Tensor dtype shape
  -> Tensor dtype shape
layerNorm LayerNorm {..} = Torch.Static.Native.layerNorm @normalizedShape
  (toDependent layerNormWeight)
  (toDependent layerNormBias)
  layerNormEps

data LinearSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) = LinearSpec
  deriving (Show, Eq)

data Linear (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) =
  Linear { weight :: Parameter dtype '[inputFeatures, outputFeatures]
         , bias :: Parameter dtype '[outputFeatures]
         } deriving (Show, Generic)

linear
  :: forall dtype (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat])
   . ( CheckBroadcast (CheckMatMul
                         shape
                         '[inputFeatures, outputFeatures]
                         (ComputeMatMul
                            (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                      '[outputFeatures]
                      (ComputeBroadcast
                         (ReverseImpl
                            (CheckMatMul
                               shape
                               '[inputFeatures, outputFeatures]
                               (ComputeMatMul
                                  (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                            '[])
                         '[outputFeatures])
                    ~ shape')
  => Linear dtype inputFeatures outputFeatures
  -> Tensor dtype shape
  -> Tensor dtype shape'
linear Linear {..} input =
  add (matmul input (toDependent weight)) (toDependent bias)

makeIndependent :: Tensor dtype shape -> IO (Parameter dtype shape)
makeIndependent t = Parameter <$> A.makeIndependent (toDynamic t)

instance A.Parameterized (Linear dtype inputFeatures outputFeatures)

instance (KnownDType dtype, KnownNat inputFeatures, KnownNat outputFeatures) => A.Randomizable (LinearSpec dtype inputFeatures outputFeatures) (Linear dtype inputFeatures outputFeatures) where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data MLPSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat) = MLPSpec

data MLP (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat) =
  MLP { layer0 :: Linear dtype inputFeatures hiddenFeatures
      , layer1 :: Linear dtype hiddenFeatures hiddenFeatures
      , layer2 :: Linear dtype hiddenFeatures outputFeatures
      } deriving (Show, Generic)

instance A.Parameterized (MLP dtype inputFeatures outputFeatures hiddenFeatures)

instance (KnownDType dtype, KnownNat inputFeatures, KnownNat outputFeatures, KnownNat hiddenFeatures) => A.Randomizable (MLPSpec dtype inputFeatures outputFeatures hiddenFeatures) (MLP dtype inputFeatures outputFeatures hiddenFeatures) where
  sample MLPSpec =
    MLP <$> A.sample LinearSpec <*> A.sample LinearSpec <*> A.sample LinearSpec

mlp
  :: MLP dtype inputFeatures outputFeatures hiddenFeatures
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
mlp MLP {..} = linear layer2 . tanh . linear layer1 . tanh . linear layer0

model
  :: MLP dtype inputFeatures outputFeatures hiddenFeatures
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
model = (sigmoid .) . mlp

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

xor
  :: forall dtype batchSize
   . Tensor dtype '[batchSize, 2]
  -> Tensor dtype '[batchSize]
xor t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
 where
  a = select @1 @0 t
  b = select @1 @1 t

main = do
  let numIters = 100000
  init    <- A.sample (MLPSpec :: MLPSpec 'D.Float 2 1 4)
  trained <- foldLoop init numIters $ \state i -> do
    input <-
      toDType @D.Float
      .   gt (0.5 :: Tensor 'D.Float '[])
      <$> rand @D.Float @'[256, 2]

    let expected_output = xor input
    let actual_output   = squeezeAll . model state $ input
    let loss            = mse_loss actual_output expected_output

    let flat_parameters = A.flattenParameters state
    let gradients       = A.grad (toDynamic loss) flat_parameters

    when (i `mod` 2500 == 0) (print loss)

    new_flat_parameters <- mapM A.makeIndependent
      $ A.sgd 1e-1 flat_parameters gradients
    return $ A.replaceParameters state new_flat_parameters
  print trained
