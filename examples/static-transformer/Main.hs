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
import           Data.HList
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Functional     hiding ( linear )
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional               as D
import qualified Torch.TensorFactories         as D


--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data Activation (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  Activation
    :: forall dtype device
     . { unActivation :: forall shape . Tensor device dtype shape -> Tensor device dtype shape }
    -> Activation dtype device

instance Show (Activation dtype device) where
  show _ = mempty

instance A.Parameterized (Activation dtype device) where
  flattenParameters _ = []
  replaceOwnParameters = return

data MultiheadAttentionSpec (embedDim :: Nat) (numHeads :: Nat)
                            (dtype :: D.DType)
                            (device :: (D.DeviceType, Nat))
 where
  MultiheadAttentionSpec
    :: { mhDropoutProbSpec :: Double }
    -> MultiheadAttentionSpec embedDim numHeads dtype device
 deriving (Show, Eq)

data MultiheadAttention (embedDim :: Nat) (numHeads :: Nat)
                        (dtype :: D.DType)
                        (device :: (D.DeviceType, Nat))
 where
  MultiheadAttention
    :: { mhInProj  :: Linear embedDim (embedDim * 3) dtype device
       , mhOutProj :: Linear embedDim embedDim       dtype device
       , mhDropout :: Dropout
       }
    -> MultiheadAttention embedDim numHeads dtype device
 deriving (Show, Generic)

multiheadAttention
  :: forall embedDim numHeads seqLen batchSize headDim dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
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
  let q :. k :. v :. HNil = chunk @3 @2 . linear mhInProj $ input
  attnWeights <-
    Torch.Typed.NN.dropout mhDropout train
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
    toDType @D.Bool . triu 1 $ ones @'[seqLen, seqLen] @D.Int8 @device

multiheadAttention'
  :: forall embedDim numHeads seqLen batchSize headDim dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
     , KnownDType dtype
     , dtype ~ SumDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     , MatMulDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => MultiheadAttention embedDim numHeads dtype device
  -> Bool
  -> Kleisli
       IO
       ( Tensor device 'D.Bool '[seqLen, batchSize]
       , Tensor device dtype   '[seqLen, batchSize, embedDim]
       )
       ( Tensor device dtype '[seqLen, batchSize, embedDim]
       , Tensor device dtype '[batchSize, seqLen, seqLen]
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
      ^>> Kleisli (Torch.Typed.NN.dropout mhDropout train)
   where
    dotProduct =
      let scaling = pow (-1 / 2 :: Double) (fromInteger . natVal $ Proxy @headDim) :: Tensor device dtype '[]
      in  ((f >>^ mul scaling) *** (f >>^ transpose @1 @2)) >>^ uncurry matmul
    maskFutureTimestamps =
      let futureTimestampMask = toDType @D.Bool . triu 1 $ ones @'[seqLen, seqLen] @D.Int8 @device
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
    let factor = pow (-1 :: Double) (fromInteger . natVal $ Proxy @numHeads) :: Tensor device dtype '[]
    in  reshape @'[batchSize, numHeads, seqLen, seqLen]
          ^>> sumDim @1
          ^>> mul factor
          ^>> returnA
  f = reshape @'[seqLen, batchSize * numHeads, headDim] ^>> transpose @0 @1 ^>> returnA

instance A.Parameterized (MultiheadAttention embedDim numHeads dtype device)  

instance ( KnownNat embedDim
         , KnownNat numHeads
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
      <*> A.sample (DropoutSpec mhDropoutProbSpec)

data TransformerLMLayerSpec (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
                            (dtype :: D.DType)
                            (device :: (D.DeviceType, Nat))
 where
  TransformerLMLayerSpec
    :: forall embedDim numHeads ffnDim dtype device
     . { tMHDropoutProbSpec   :: Double
       , tAttnDropoutProbSpec :: Double
       , tLNEpsSpec           :: Double
       , tMLPDropout0ProbSpec :: Double
       , tMLPDropout1ProbSpec :: Double
       , tMLPActivation0Spec  :: Activation dtype device
       , tMLPActivation1Spec  :: Activation dtype device
       }
    -> TransformerLMLayerSpec embedDim numHeads ffnDim dtype device
 deriving (Show)

data TransformerLMLayer (embedDim :: Nat) (numHeads :: Nat) (ffnDim :: Nat)
                        (dtype :: D.DType)
                        (device :: (D.DeviceType, Nat))
 where
  TransformerLMLayer
    :: forall embedDim numHeads ffnDim dtype device
     . { tAttn        :: MultiheadAttention embedDim numHeads dtype device
       , tAttnDropout :: Dropout
       , tLN0         :: LayerNorm '[embedDim] dtype device
       , tLN1         :: LayerNorm '[embedDim] dtype device
       , tMLP         :: TransformerLMMLP embedDim ffnDim dtype device
       }
    -> TransformerLMLayer embedDim numHeads ffnDim dtype device
 deriving (Show, Generic)

transformerLMLayer
  :: forall numHeads ffnDim embedDim headDim seqLen batchSize dtype device
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
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
  (attn, _) <- multiheadAttention tAttn train keyPaddingMask input
  x         <- Torch.Typed.NN.dropout tAttnDropout train attn
  let x' = Torch.Typed.NN.layerNorm tLN0 (x `add` input)
  x''       <- transformerLMMLP tMLP train x'
  return $ Torch.Typed.NN.layerNorm tLN1 (x'' `add` x')

transformerLMLayer'
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
  -> Kleisli
       IO
       ( Tensor device 'D.Bool '[seqLen, batchSize]
       , Tensor device dtype   '[seqLen, batchSize, embedDim]
       )
       (Tensor device dtype '[seqLen, batchSize, embedDim])
transformerLMLayer' TransformerLMLayer {..} train =
  (arr snd &&& attn)
    >>> uncurry add
    ^>> Torch.Typed.NN.layerNorm tLN0
    ^>> (id &&& transformerLMMLP' tMLP train)
    >>^ uncurry add
    >>^ Torch.Typed.NN.layerNorm tLN1
 where
  attn = multiheadAttention' tAttn train >>> arr fst >>> Kleisli
    (Torch.Typed.NN.dropout tAttnDropout train)

instance A.Parameterized (TransformerLMLayer embedDim numHeads ffnDim dtype device)

instance ( KnownNat embedDim
         , KnownNat numHeads
         , KnownNat ffnDim
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)
                    (TransformerLMLayer     embedDim numHeads ffnDim dtype device)
 where
  sample TransformerLMLayerSpec {..} =
    let mhDropoutProbSpec = tMHDropoutProbSpec
        dropoutProbSpec   = tAttnDropoutProbSpec
        layerNormEpsSpec  = tLNEpsSpec
        tDropout0ProbSpec = tMLPDropout0ProbSpec
        tDropout1ProbSpec = tMLPDropout1ProbSpec
        tActivation0Spec  = tMLPActivation0Spec
        tActivation1Spec  = tMLPActivation1Spec
    in  TransformerLMLayer
          <$> A.sample MultiheadAttentionSpec {..}
          <*> A.sample DropoutSpec {..}
          <*> A.sample LayerNormSpec {..}
          <*> A.sample LayerNormSpec {..}
          <*> A.sample TransformerLMMLPSpec {..}

data TransformerLMMLPSpec (embedDim :: Nat) (ffnDim :: Nat)
                          (dtype :: D.DType)
                          (device :: (D.DeviceType, Nat))
 where
  TransformerLMMLPSpec
    :: forall embedDim ffnDim dtype device
     . { tDropout0ProbSpec :: Double
       , tDropout1ProbSpec :: Double
       , tActivation0Spec :: Activation dtype device
       , tActivation1Spec :: Activation dtype device
       }
    -> TransformerLMMLPSpec embedDim ffnDim dtype device

data TransformerLMMLP (embedDim :: Nat) (ffnDim :: Nat)
                      (dtype :: D.DType)
                      (device :: (D.DeviceType, Nat))
 where
  TransformerLMMLP
    :: forall embedDim ffnDim dtype device
     . { tLinear0     :: Linear embedDim ffnDim dtype device
       , tLinear1     :: Linear ffnDim embedDim dtype device
       , tDropout0    :: Dropout
       , tDropout1    :: Dropout
       , tActivation0 :: Activation dtype device
       , tActivation1 :: Activation dtype device
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
  Torch.Typed.NN.dropout tDropout1 train
    .   unActivation tActivation1
    .   linear tLinear1
    =<< Torch.Typed.NN.dropout tDropout0 train
    .   unActivation tActivation0
    .   linear tLinear0
    =<< pure input

transformerLMMLP'
  :: forall embedDim ffnDim seqLen batchSize dtype device
   . TransformerLMMLP embedDim ffnDim dtype device
  -> Bool
  -> Kleisli
       IO
       (Tensor device dtype '[seqLen, batchSize, embedDim])
       (Tensor device dtype '[seqLen, batchSize, embedDim])
transformerLMMLP' TransformerLMMLP {..} train =
  linear tLinear0
    ^>> unActivation tActivation0
    ^>> Kleisli (Torch.Typed.NN.dropout tDropout0 train)
    >>> linear tLinear1
    ^>> unActivation tActivation1
    ^>> Kleisli (Torch.Typed.NN.dropout tDropout1 train)

instance A.Parameterized (TransformerLMMLP embedDim ffnDim dtype device)

instance ( KnownNat embedDim
         , KnownNat ffnDim
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
      <*> A.sample (DropoutSpec tDropout0ProbSpec)
      <*> A.sample (DropoutSpec tDropout1ProbSpec)
      <*> pure tActivation0Spec
      <*> pure tActivation1Spec

data FoldLayers = FoldLayers { foldLayersTrain :: Bool }

instance ( 1 <= numHeads
         , embedDim ~ (headDim * numHeads)
         , Mod (embedDim * 3) 3 ~ 0
         , Div (embedDim * 3) 3 ~ embedDim
         , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
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

data FoldLayers' = FoldLayers' { foldLayersTrain' :: Bool }

instance ( 1 <= numHeads
         , embedDim ~ (headDim * numHeads)
         , Mod (embedDim * 3) 3 ~ 0
         , Div (embedDim * 3) 3 ~ embedDim
         , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
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
         FoldLayers'
         (TransformerLMLayer embedDim numHeads ffnDim dtype device)
         (Kleisli IO (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim]) (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim]))
 where
  apply FoldLayers' {..} layer = arr fst &&& transformerLMLayer' layer foldLayersTrain'

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
  => Embedding ('Just paddingIdx) numEmbeds embedDim dtype device
  -> Embedding 'Nothing           2048      embedDim dtype device
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
                    . toDType @D.Int64
                    . linspace @seqLen (0 :: Int)
                    $ natValI @(seqLen - 1)
  x <- Torch.Typed.NN.dropout dropout train (src `add` positions)
  let keyPaddingMask = srcTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[])
  (_, x') <- hfoldrM (FoldLayers train) (keyPaddingMask, x) layers
  return x'

getHidden'
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
     , HFoldrM'
         IO
         FoldLayers'
         (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDevice device
     )
  => Embedding ('Just paddingIdx) numEmbeds embedDim dtype device
  -> Embedding 'Nothing           2048      embedDim dtype device
  -> Dropout
  -> Bool
  -> HList (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
  -> Kleisli IO (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[seqLen, batchSize, embedDim])
getHidden' embedding posEmbedding dropout train layers =
  transpose @0 @1
    ^>> (mkKeyPaddingMask &&& mkInput)
    >>> hfoldrM' (FoldLayers' train) layers
    >>^ snd
 where
  mkKeyPaddingMask =
    arr (==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[]))
  mkInput = embed embedding ^>> add positions ^>> Kleisli (Torch.Typed.NN.dropout dropout train)
  positions =
    expand @'[seqLen, batchSize, embedDim] True
      . unsqueeze @1
      . embed posEmbedding
      . toDType @D.Int64
      . linspace @seqLen (0 :: Int)
      $ natValI @(seqLen - 1)

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
    :: forall
         numAttnLayers
         numHeads
         ffnDim
         paddingIdx
         numEmbeds
         embedDim
         seqLen
         dtype
         device
     . { tDropoutProbSpec :: Double
       , tLayerSpec :: TransformerLMLayerSpec embedDim numHeads ffnDim dtype device
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
    :: forall
         numAttnLayers
         numHeads
         ffnDim
         paddingIdx
         numEmbeds
         embedDim
         seqLen
         dtype
         device
     . { tEmbedding    :: Embedding ('Just paddingIdx) numEmbeds embedDim dtype device
       , tPosEmbedding :: Embedding 'Nothing           2048      embedDim dtype device
       , tDropout      :: Dropout
       , tLayers       :: HList (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
       , tProj         :: Linear embedDim seqLen dtype device
       }
    -> TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
 deriving (Generic)

instance (A.Parameterized (HList (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))))
  => A.Parameterized (TransformerLM numAttnLayers
                                    numHeads
                                    ffnDim
                                    paddingIdx
                                    numEmbeds
                                    embedDim
                                    seqLen
                                    dtype
                                    device)

instance ( paddingIdx <= numEmbeds
         , 1 <= numEmbeds - paddingIdx
         , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
         , KnownNat ffnDim
         , KnownNat paddingIdx
         , KnownNat numEmbeds
         , KnownNat embedDim
         , KnownNat seqLen
         , HReplicate'
             numAttnLayers
             (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)
             (HReplicateR
               numAttnLayers
               (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device))
         , A.Randomizable
             (HList
               (HReplicateR
                 numAttnLayers
                 (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)))
             (HList
               (HReplicateR
                 numAttnLayers
                 (TransformerLMLayer embedDim numHeads ffnDim dtype device)))
         , KnownDType dtype
         , RandDTypeIsValid device dtype
         , KnownDevice device
         )
  => A.Randomizable (TransformerLMSpec numAttnLayers
                                       numHeads
                                       ffnDim
                                       paddingIdx
                                       numEmbeds
                                       embedDim
                                       seqLen
                                       dtype
                                       device)
                    (TransformerLM     numAttnLayers
                                       numHeads
                                       ffnDim
                                       paddingIdx
                                       numEmbeds
                                       embedDim
                                       seqLen
                                       dtype
                                       device)
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
                    (TransformerLMLayerSpec embedDim numHeads ffnDim dtype device)
                )
            )
      <*> A.sample LinearSpec

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

logits'
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
     , HFoldrM'
         IO
         FoldLayers'
         (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim])
         (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDevice device
     )
  => TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
  -> Bool
  -> Kleisli IO (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, seqLen])
logits' TransformerLM {..} train =
  getHidden' @numAttnLayers @numHeads @ffnDim tEmbedding
                                              tPosEmbedding
                                              tDropout
                                              train
                                              tLayers
    >>^ transpose @0 @1
    >>^ linear tProj

crossEntropyLoss
  :: forall paddingIdx batchSize seqLen dtype device
   . ( KnownNat paddingIdx
     , KnownNat batchSize
     , KnownNat seqLen
     , KnownDType dtype
     , KnownDevice device
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype '[batchSize, seqLen, seqLen]
  -> Tensor device 'D.Int64 '[batchSize, seqLen]
  -> Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @seqLen @'[seqLen]
    ones
    (natValI @paddingIdx)
    (logSoftmax @1 prediction)
    target

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

type Model device
  = TransformerLM
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      NumEmbeds
      EmbedDim
      SeqLen
      'D.Float
      device

type ModelSpec device
  = TransformerLMSpec
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      NumEmbeds
      EmbedDim
      SeqLen
      'D.Float
      device

data Data

type BatchSize = 1

train
  :: forall (device :: (D.DeviceType, Nat))
   . ( KnownDevice device
     , RandDTypeIsValid device 'D.Float
     , StandardFloatingPointDTypeValidation device 'D.Float
     , MatMulDTypeIsValid device 'D.Float
     , BasicArithmeticDTypeIsValid device 'D.Float
     , SumDTypeIsValid device 'D.Float
     , ComparisonDTypeIsValid device 'D.Float
     , ComparisonDTypeIsValid device 'D.Int64
     )
  => Int
  -> IO ()
train numIters = do
  init          <- A.sample spec :: IO (Model device)
  (_trained, _) <- foldLoop (init, undefined) numIters $ \(state, _) i -> do
    trainingLoss <- computeLoss @BatchSize state True undefined undefined
    let flat_parameters = A.flattenParameters state
    let gradients       = A.grad (toDynamic trainingLoss) flat_parameters
    new_flat_parameters <- mapM A.makeIndependent
      $ A.sgd 1e-01 flat_parameters gradients
    return (A.replaceParameters state new_flat_parameters, undefined)
  return ()
 where
  spec :: ModelSpec device
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
    => Model device
    -> Bool
    -> [Int]
    -> Data
    -> IO (Tensor device 'D.Float '[])
  computeLoss state train _indexes _data = do
    let input  = (undefined :: Tensor device 'D.Int64 '[batchSize, SeqLen])
        target = (undefined :: Tensor device 'D.Int64 '[batchSize, SeqLen])
    prediction <- logits state train input
    return $ crossEntropyLoss @PaddingIdx prediction (toDevice target)

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu"    -> train @'( 'D.CPU, 0)  1
    Right "cuda:0" -> train @'( 'D.CUDA, 0) 1
    _              -> error "Don't know what to do or how."
