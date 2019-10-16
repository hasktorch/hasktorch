{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}

module Main where

import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.Maybe
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Exts
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
import           Torch.Static.Native
import           Torch.Static.Factories
import           Torch.Static.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D
import qualified Image                         as I

type NoStrides = '(1, 1)
type NoPadding = '(0, 0)

type KernelSize = '(2, 2)
type Strides = '(2, 2)

data CNNSpec (dtype :: D.DType)
  = CNNSpec deriving (Show, Eq)

data CNN (dtype :: D.DType)
 where
  CNN
    :: forall dtype
     . { conv0 :: Conv2d dtype 1  20 5 5
       , conv1 :: Conv2d dtype 20 50 5 5
       , fc0   :: Linear dtype (4*4*50) 500
       , fc1   :: Linear dtype 500      10
       }
    -> CNN dtype
 deriving (Show, Generic)

cnn
  :: forall dtype batchSize
   . _
  => CNN dtype
  -> Tensor dtype '[batchSize, I.DataDim]
  -> Tensor dtype '[batchSize, I.ClassDim]
cnn CNN {..} =
  Torch.Static.NN.linear fc1
    . relu
    . Torch.Static.NN.linear fc0
    . reshape @'[batchSize, 4*4*50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Static.NN.conv2d @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Static.NN.conv2d @NoStrides @NoPadding conv0
    . unsqueeze @1
    . reshape @'[batchSize, I.Rows, I.Cols]

instance A.Parameterized (CNN dtype)

instance (KnownDType dtype)
  => A.Randomizable (CNNSpec dtype)
                    (CNN     dtype)
 where
  sample CNNSpec =
    CNN
      <$> A.sample (Conv2dSpec @dtype @1  @20 @5 @5)
      <*> A.sample (Conv2dSpec @dtype @20 @50 @5 @5)
      <*> A.sample (LinearSpec @dtype @(4*4*50) @500)
      <*> A.sample (LinearSpec @dtype @500      @10)

fakeRandom :: Int -> (Int, Float)
fakeRandom i = (j, ans)
 where
  j   = 7 * i `mod` 101
  ans = (-5.5 + (fromIntegral $ (j - 1) `mod` 10 + 1)) / 9.0

iterate' :: (a -> (a, b)) -> a -> [b]
iterate' f a = let (a', b) = f a in b : iterate' f a'

mkFakeRandomCNN :: IO (CNN D.Float)
mkFakeRandomCNN = do
  conv0Weight <- makeIndependent . reshape @_ @_ @'[20*1*5*5] . fromJust . fromList . take (20*1*5*5) . iterate' fakeRandom $ 13 :: IO (Parameter D.Float '[20, 1, 5, 5])
  conv0Bias <- makeIndependent . fromJust . fromList . take 20 . iterate' fakeRandom $ 13 :: IO (Parameter D.Float '[20])
  let conv0 = Conv2d { conv2dWeight = conv0Weight
                     , conv2dBias = conv0Bias
                     } :: Conv2d D.Float 1 20 5 5
  conv1Weight <- makeIndependent . reshape @_ @_ @'[50*20*5*5] . fromJust . fromList . take (50*20*5*5) . iterate' fakeRandom $ 82 :: IO (Parameter D.Float '[50, 20, 5, 5])
  conv1Bias <- makeIndependent . fromJust . fromList . take 50 . iterate' fakeRandom $ 82 :: IO (Parameter D.Float '[50])
  let conv1 = Conv2d { conv2dWeight = conv1Weight
                     , conv2dBias = conv1Bias
                     } :: Conv2d D.Float 20 50 5 5
  fc0Weight <- makeIndependent . reshape @_ @_ @'[500*4*4*50] . fromJust . fromList . take (500*4*4*50) . iterate' fakeRandom $ 19 :: IO (Parameter D.Float '[500, 4*4*50])
  fc0Bias <- makeIndependent . fromJust . fromList . take 500 . iterate' fakeRandom $ 19 :: IO (Parameter D.Float '[500])
  let fc0 = Linear { linearWeight = fc0Weight
                   , linearBias = fc0Bias
                   } :: Linear D.Float (4*4*50) 500
  fc1Weight <- makeIndependent . reshape @_ @_ @'[10*500] . fromJust . fromList . take (10*500) . iterate' fakeRandom $ 19 :: IO (Parameter D.Float '[10, 500])
  fc1Bias <-  makeIndependent . fromJust . fromList . take 10 . iterate' fakeRandom $ 19 :: IO (Parameter D.Float '[10])
  let fc1 = Linear { linearWeight = fc1Weight
                   , linearBias = fc1Bias
                   } :: Linear D.Float 500 10
  return $ CNN { conv0 = conv0
               , conv1 = conv1
               , fc0 = fc0
               , fc1 = fc1
               }

mkFakeRandomInput :: Tensor D.Float '[1, I.DataDim]
mkFakeRandomInput = reshape @_ @_ @'[1*I.DataDim] . fromJust . fromList . take (1*natValI @I.DataDim) . iterate' fakeRandom $ 17

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

type BatchSize = 64
type TestBatchSize = 1000

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
  backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
  let backend = case backend' of
        Right "CUDA" -> "CUDA"
        _            -> "CPU"
      (numIters, printEvery) = (1000000, 250)
  (trainingData, testData) <- I.initMnist
  init                     <- A.sample (CNNSpec @'D.Float)
  init' <- A.replaceParameters init <$> traverse
    (A.makeIndependent . toBackend backend . A.toDependent)
    (A.flattenParameters init)
  (trained, _) <-
    foldLoop (init', randomIndexes (I.length trainingData)) numIters
      $ \(state, idxs) i -> do
          let (indexes, nextIndexes) =
                (take (natValI @I.DataDim) idxs, drop (natValI @I.DataDim) idxs)
          (trainingLoss, _) <- computeLossAndErrorRate @BatchSize backend
                                                                  state
                                                                  True
                                                                  indexes
                                                                  trainingData
          let flat_parameters = A.flattenParameters state
          let gradients       = A.grad (toDynamic trainingLoss) flat_parameters
          when (i `mod` printEvery == 0)
            $ case someNatVal (fromIntegral $ I.length testData) of
                Just (SomeNat (Proxy :: Proxy testSize)) -> do
                  (testLoss, testError) <-
                    computeLossAndErrorRate @(Min TestBatchSize testSize)
                      backend
                      state
                      False
                      (randomIndexes (I.length testData))
                      testData
                  printLosses i trainingLoss testLoss testError
                _ -> print "Cannot get the size of the test dataset"

          new_flat_parameters <- mapM A.makeIndependent
            $ A.sgd 1e-02 flat_parameters gradients
          return (A.replaceParameters state new_flat_parameters, nextIndexes)
  print trained
 where
  computeLossAndErrorRate
    :: forall n
     . (KnownNat n)
    => String
    -> CNN 'D.Float
    -> Bool
    -> [Int]
    -> I.MnistData
    -> IO (Tensor 'D.Float '[], Tensor 'D.Float '[])
  computeLossAndErrorRate backend state train indexes data' = do
    let input  = toBackend backend $ I.getImages @n data' indexes
        target = toBackend backend $ I.getLabels @n data' indexes
        result = cnn state input
    return (crossEntropyLoss backend result target, errorRate result target)
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
