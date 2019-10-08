{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Control.Exception.Safe         (try, SomeException(..))
import           GHC.Generics
import           GHC.TypeLits

import           Torch.Static
import           Torch.Static.Native     hiding ( linear )
import           Torch.Static.Factories
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D
import qualified Image                         as I
import           System.Random
import           System.Environment
import           Data.Proxy

--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------


newtype Parameter dtype shape = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter dtype shape -> Tensor dtype shape
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t

instance A.Parameterized (Parameter dtype shape) where
  flattenParameters (Parameter x) = [x]
  replaceOwnParameters _ = Parameter <$> A.nextParameter

data LinearSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) = LinearSpec
  deriving (Show, Eq)

data Linear (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) =
  Linear { weight :: Parameter dtype '[inputFeatures, outputFeatures]
         , bias :: Parameter dtype '[outputFeatures]
         } deriving (Show, Generic)

linear
  :: Linear dtype inputFeatures outputFeatures
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
linear Linear {..} input =
  add (mm input (toDependent weight)) (toDependent bias)

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
mlp MLP {..} = linear layer2 . relu . linear layer1 . relu . linear layer0

model
  :: KnownDType dtype
  => MLP dtype inputFeatures outputFeatures hiddenFeatures
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
model = (softmax @1 .) . mlp

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

type BatchSize = 64

randomIndexes :: Int -> [Int]
randomIndexes size = map (`mod` size) $ randoms seed
  where
    seed = mkStdGen 123

toBackend :: String -> Tensor dtype shape -> Tensor dtype shape
toBackend backend t =
  case backend of
    "CUDA" -> toCUDA t
    _ -> toCPU t

main = do
  backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
  let backend = 
        case backend' of
          Right "CUDA" -> "CUDA"
          _ -> "CPU"
  let numIters = 10000
  (trainingData, testData) <- I.initMnist
  init    <- A.sample (MLPSpec :: MLPSpec 'D.Float I.DataDim I.ClassDim 128)
  
  
  (trained,_)  <- foldLoop (init,randomIndexes (I.length trainingData)) numIters $ \(state, idxs) i -> do
    let (indexes,nextIndexes) = (take (natValI @I.DataDim) idxs, drop (natValI @I.DataDim) idxs)
    let input           = toBackend backend $ I.getImages @BatchSize trainingData indexes 
    let expected_output = toBackend backend $ I.getLabels @BatchSize trainingData indexes
    let actual_output   = model state $ input
    let loss            = mse_loss actual_output expected_output

    let flat_parameters = A.flattenParameters state
    let gradients       = A.grad (toDynamic loss) flat_parameters

    when (i `mod` 250 == 0) (print loss)

    new_flat_parameters <- mapM A.makeIndependent
      $ A.sgd 1e-1 flat_parameters gradients
    return $ (A.replaceParameters state new_flat_parameters, nextIndexes)

  case someNatVal (fromIntegral $ I.length testData)of
    Just (SomeNat (Proxy :: Proxy numTest)) -> do
      let test_data = toBackend backend $ I.getImages @numTest testData [0..]
          test_label = toBackend backend $ I.getLabels @numTest testData [0..]
      putStrLn $ "Learning loss: the number of test is " ++ show (natValI @numTest) ++ "."
      print $ mse_loss (model trained test_data) test_label
    _ -> print "Can not get the number of test"
