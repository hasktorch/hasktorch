{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
module Main where

import Prelude
import Utils
import LeNet (LeNet)

import Control.Arrow
import Control.Monad
import Data.Maybe
import Data.Either
import Data.List
import Data.Vector (Vector)
import Data.HashMap.Strict (HashMap)
import Text.Printf
import Numeric.Backprop
import System.IO (hFlush, stdout)
import System.IO.Unsafe (unsafePerformIO)
import qualified Numeric.Backprop as Bp
import qualified Data.Vector as V
import qualified Data.HashMap.Strict as HM

#ifdef CUDA
import Torch.Cuda.Double as Torch hiding (Sum)
import qualified Torch.Cuda.Double.Storage as S
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
import Foreign
import Torch.FFI.THC.State
import qualified Torch.Cuda.Double.NN.Conv2d as Conv2d
#else
import Torch.Double as Torch hiding (Sum)
import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
#endif

import Torch.Models.Vision.LeNet (newLeNet, lenet)
import Torch.Data.Loaders.Cifar10
import Torch.Data.Loaders.Internal
import Torch.Data.Loaders.RGBVector (Normalize(..))
import qualified Torch.Models.Vision.LeNet as LeNet



infer
  :: LeNet
  -> Tensor '[3, 32, 32]
  -> Category
infer net

  -- cast from Integer to 'Torch.Data.Loaders.Cifar10.Category'
  = toEnum . fromIntegral

  -- Unbox the LongTensor '[1] to get 'Integer'
  . getindex

  -- argmax the output Tensor '[10] distriubtion. Returns LongTensor '[1]
  . maxIndex1d

#ifndef DEBUG
  -- take an input tensor and run 'lenet' with the model (undefined is the
  -- learning rate, which we can ignore)
  . evalBP2 (lenet undefined) net

#else
  . foo

 where
  foo x
    -- take an input tensor and run 'lenet' with the model (undefined is the
    -- learning rate, which we can ignore)
    = unsafePerformIO $ do
        let x' = evalBP2 (lenet undefined) net x
        pure x'
#endif

main :: IO ()
main = do
  g <- seedAll
  ltest <- prepdata . V.take 200 <$> cifar10set g default_cifar_path Test

  net0 :: LeNet
    <- newLeNet =<< newRNG
  print net0

  putStr "\nInitial Holdout:\t"
  hFlush stdout

  test <-
    if all isRight (V.map snd ltest)
    then pure $ V.map (second (fromRight (error "should already be instantiated"))) ltest
    else do
      let l = fromIntegral (length ltest) :: Float
      V.mapM getdata ltest

  print $ V.head test
  hFlush stdout

  let
    testX = V.toList $ fmap snd test
    testY = V.toList $ fmap fst test
    preds = map (infer net0) testX
    acc = genericLength (filter id $ zipWith (==) preds testY) / genericLength testY

  printf "\n"
  printf ("[test accuracy: %.1f%% / %d]\tAll same? %s")
    (acc * 100 :: Float)
    (length testY)
    (if all (== head preds) preds then show (head preds) else "No.")

  hFlush stdout

  -- report
  let
    test' :: Vector (Category, Tensor '[3,32,32])
    test' = V.map (second (fromRight undefined)) (fmap (second Right) test)

    cathm :: [(Category, [Tensor '[3, 32, 32]])]
    cathm = HM.toList $ HM.fromListWith (++) $ V.toList (second (:[]) <$> test)

  forM_ cathm $ \(y, xs) -> do
    let
      preds = map (infer net0) xs
      correct = length (filter (==y) preds)
      acc = fromIntegral correct / genericLength xs :: Float

    printf "\n[%s]: %.2f%% (%d / %d)" (show y) (acc*100) correct (length xs)
    hFlush stdout


-------------------------------------------------------------------------------
-- Sanity check tests

#ifdef DEBUG
-- There is a bug here having to do with CUDA.
loadtest :: IO ()
loadtest = do
  g <- MWC.initialize (V.singleton 42)
  ltrain <- fmap (second Left) . V.take 5000 <$> cifar10set g default_cifar_path Train
  forM_ ltrain $ \(_, y) -> case y of
    Left x -> insanity x
    Right x -> pure x


insanity :: FilePath -> IO (Tensor '[3, 32, 32])
insanity f = go 0
 where
  go x =
    runExceptT (rgb2torch ZeroToOne f) >>= \case
      Left s -> throwString s
      Right t -> do
#ifdef CUDA
        CPU.tensordata (copyDouble t) >>= \rs -> do
#else
        tensordata t >>= \rs -> do
#endif
          let
              oob = filter (\x -> x < 0 || x > 1) rs
              oox = filter (<= 2) oob
          if not (null oob)
          then throwString (show (oob, oox))
          else
            if not (all (== 0) rs)
            then pure t
            else if x == 10
              then throwString $ f ++ ": 10 retries -- failing on all-zero tensor"
              else do
                print $ f ++ ": retrying from " ++ show x
                threadDelay 1000000
                go (x+1)
#endif


