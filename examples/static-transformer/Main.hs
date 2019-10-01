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

import           Prelude                 hiding ( tanh )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.List                      ( foldl'
                                                , scanl'
                                                , intersperse
                                                )
import           Data.Reflection
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
-- Multi-Layer Perceptron (MLP)
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
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim], Tensor dtype '[batchSize, seqLen, seqLen])
multiheadAttention MultiheadAttention {..} input = do
  let q :. k :. v :. HNil = chunk @3 @2 . linear mhInProj $ input
      scaling     = pow (-0.5) (natValI @headDim) :: Tensor dtype '[]
      f           = transpose @0 @1 . reshape @'[seqLen, batchSize * numHeads, headDim]
      attnWeights = (f . mul scaling $ q) `matmul` (transpose  @1 @2 . f $ k)
  -- TODO: mask future timesteps
  -- TODO: apply key padding mask
  attnWeights' <- Main.dropout mhDropout . softmax @2 $ attnWeights
  let attn =
        linear mhOutProj
          . reshape @'[seqLen, batchSize, embedDim]
          . transpose @0 @1
          . matmul attnWeights'
          . f
          $ v
      avgAttnWeights =
        mul (pow (-1) (natValI @numHeads) :: Tensor dtype '[])
          . sumDim @1
          . reshape @'[batchSize, numHeads, seqLen, seqLen]
          $ attnWeights'
  return (attn, avgAttnWeights)

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
    :: { tSelfAttention :: MultiheadAttention dtype embedDim numHeads
       , tFc0 :: Linear dtype embedDim ffnDim
       , tFc1 :: Linear dtype ffnDim embedDim
       , tLn0 :: LayerNorm dtype '[embedDim]
       , tLn1 :: LayerNorm dtype '[embedDim]
       , tDropout :: Dropout
       , tActivation :: forall shape . Tensor dtype shape -> Tensor dtype shape
       }
    -> TransformerLMLayer dtype embedDim numHeads ffnDim

transformerLMLayer
  :: forall dtype embedDim numHeads seqLen batchSize ffnDim headDim
   . ( 1 <= numHeads
     , embedDim ~ (headDim * numHeads)
     , Mod (embedDim * 3) 3 ~ 0
     , Div (embedDim * 3) 3 ~ embedDim
     , KnownDType dtype
     , All KnownNat [embedDim, numHeads, seqLen, batchSize, headDim]
     , EndsWith '[seqLen, batchSize, embedDim] '[embedDim]
     )
  => TransformerLMLayer dtype embedDim numHeads ffnDim
  -> Tensor dtype '[seqLen, batchSize, embedDim]
  -> IO (Tensor dtype '[seqLen, batchSize, embedDim])
transformerLMLayer TransformerLMLayer {..} input = do
  (attn, _) <- multiheadAttention tSelfAttention input
  x         <- Main.layerNorm tLn0 . add input <$> Main.dropout tDropout attn
  x'        <- Main.dropout tDropout . tActivation . linear tFc0 $ x
  x''       <- Main.dropout tDropout . tActivation . linear tFc1 $ x'
  return $ Main.layerNorm tLn1 (x'' `add` x)

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
