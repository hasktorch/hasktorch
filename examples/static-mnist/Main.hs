{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe
import           System.Random

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
import qualified Image                         as I

--------------------------------------------------------------------------------
-- MNIST
--------------------------------------------------------------------------------

data MLPSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat) = MLPSpec

data MLP (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat) =
  MLP { layer0 :: Linear dtype inputFeatures hiddenFeatures
      , layer1 :: Linear dtype hiddenFeatures outputFeatures
      , dropout :: Dropout
      } deriving (Show, Generic)

instance A.Parameterized (MLP dtype inputFeatures outputFeatures hiddenFeatures)

instance (KnownDType dtype, KnownNat inputFeatures, KnownNat outputFeatures, KnownNat hiddenFeatures) => A.Randomizable (MLPSpec dtype inputFeatures outputFeatures hiddenFeatures) (MLP dtype inputFeatures outputFeatures hiddenFeatures) where
  sample MLPSpec = MLP <$> A.sample LinearSpec <*> A.sample LinearSpec

mlp
  :: MLP dtype inputFeatures outputFeatures hiddenFeatures
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
mlp MLP {..} = linear layer1 . relu . linear layer0

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

type BatchSize = 1000
type TestBatchSize = 10000
type HiddenFeatures = 500

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123

toBackend
  :: forall t . (ATen.Castable t (ForeignPtr ATen.Tensor)) => String -> t -> t
toBackend backend t = unsafePerformIO $ case backend of
  "CUDA" -> ATen.cast1 ATen.tensor_cuda t
  _      -> ATen.cast1 ATen.tensor_cpu t

crossEntropyLoss
  :: forall batchSize outputFeatures
   . (KnownNat batchSize, KnownNat outputFeatures)
  => String
  -> Tensor 'D.Float '[batchSize, outputFeatures]
  -> Tensor 'D.Int64 '[batchSize]
  -> Tensor 'D.Float '[]
crossEntropyLoss backend result target =
  nll_loss @D.ReduceMean @ 'D.Float @batchSize @outputFeatures @'[]
    (logSoftmax @1 result)
    target
    (toBackend backend ones)
    (-100)

errorRate
  :: forall batchSize outputFeatures
   . (KnownNat batchSize, KnownNat outputFeatures)
  => Tensor 'D.Float '[batchSize, outputFeatures]
  -> Tensor 'D.Int64 '[batchSize]
  -> Tensor 'D.Float '[]
errorRate result target =
  let errorCount =
          toDType @D.Float . sumAll . ne (argmax @1 @DropDim result) $ target
  in  cmul errorCount ((1.0 /) . fromIntegral $ natValI @batchSize :: Double)

main = do
  debug'   <- try (getEnv "DEBUG") :: IO (Either SomeException String)
  backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
  let backend = case backend' of
        Right "CUDA" -> "CUDA"
        _            -> "CPU"
  let debug = case debug' of
        Right "TRUE" -> True
        _            -> False
  let (numIters, printEvery) = (10000, 25)
  (trainingData, testData) <- I.initMnist
  init <- A.sample (MLPSpec @ 'D.Float @I.DataDim @I.ClassDim @HiddenFeatures)
  init' <- A.replaceParameters init <$> traverse
    (A.makeIndependent . toBackend backend . A.toDependent)
    (A.flattenParameters init)
  when debug $ print "init' is done."
  (trained, _) <-
    foldLoop (init', randomIndexes (I.length trainingData)) numIters
      $ \(state, idxs) i -> do
          let (indexes, nextIndexes) =
                (take (natValI @I.DataDim) idxs, drop (natValI @I.DataDim) idxs)
          let (trainingLoss, _) = computeLossAndErrorRate @BatchSize
                backend
                state
                indexes
                trainingData
          let flat_parameters = A.flattenParameters state
          let gradients       = A.grad (toDynamic trainingLoss) flat_parameters
          when debug $ do
            print $ "training loss:" ++ show trainingLoss
            print $ "gradients:" ++ show gradients
          when (i `mod` printEvery == 0)
            $ case someNatVal (fromIntegral $ I.length testData) of
                Just (SomeNat (Proxy :: Proxy testSize)) -> do
                  let (testLoss, testError) =
                        computeLossAndErrorRate @(Min TestBatchSize testSize)
                          backend
                          state
                          (randomIndexes (I.length testData))
                          testData
                  printLosses i trainingLoss testLoss testError
                _ -> print "Can not get the number of test"

          new_flat_parameters <- mapM A.makeIndependent
            $ A.sgd 1e-02 flat_parameters gradients
          return (A.replaceParameters state new_flat_parameters, nextIndexes)
  print trained
 where
  computeLossAndErrorRate
    :: forall n
     . (KnownNat n)
    => String
    -> MLP 'D.Float I.DataDim I.ClassDim HiddenFeatures
    -> [Int]
    -> I.MnistData
    -> (Tensor 'D.Float '[], Tensor 'D.Float '[])
  computeLossAndErrorRate backend state indexes data' =
    let input  = toBackend backend $ I.getImages @n data' indexes
        target = toBackend backend $ I.getLabels @n data' indexes
        result = mlp state input
    in  (crossEntropyLoss backend result target, errorRate result target)
  printLosses i trainingLoss testLoss testError =
    let asFloat t = D.asValue . toDynamic . toCPU $ t :: Float
    in  putStrLn
          $  "Iteration: "
          <> show i
          <> ". Training batch loss: "
          <> show (asFloat trainingLoss)
          <> ". Test loss: "
          <> show (asFloat testLoss)
          <> ". Test error-rate: "
          <> show (asFloat testError)
