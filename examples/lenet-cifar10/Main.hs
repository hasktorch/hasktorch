{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
module Main where

import Prelude

import Data.Typeable
import Data.List
import Control.Monad
import Control.Monad.Loops
import Data.Monoid
import Data.Time
import Control.Monad.IO.Class
import Text.Printf
import ListT (ListT)
import qualified ListT
import Numeric.Backprop as Bp
import Numeric.Dimensions
import System.IO.Unsafe
import GHC.TypeLits (KnownNat)
import Control.Concurrent
import qualified Prelude as P
import qualified Data.List as P ((!!))
-- import qualified Foreign.CUDA as Cuda

#ifdef CUDA
import Torch.Cuda.Double as Math hiding (Sum)
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double as Math hiding (Sum)
import qualified Torch.Long as Long
#endif

import Torch.Models.Vision.LeNet
import Torch.Data.Loaders.Cifar10
import Torch.Data.Metrics

import Control.Monad.Trans.Except
import System.IO (hFlush, stdout)

import System.Random.Shuffle (shuffleM)
-- batch dimension
-- shuffle data
-- normalize inputs

main :: IO ()
main = do
  clearScreen
  xs  <- shuffleM =<< loadTrain Nothing -- $ Just 500
  xs' <- shuffleM =<< loadTest Nothing -- $ Just 100
  net0 <- newLeNet @3 @5
  print net0
  putStrLn "Start training:"
  t0 <- getCurrentTime
  net <- epochs xs' 0.01 t0 2 xs net0
  t1 <- getCurrentTime
  printf "\nFinished training!\n"
  print net
  putStrLn "\ndone!"
 where
  loadTest  = loadData Test
  loadTrain = loadData Train
  loadData m ms = do
    t0 <- getCurrentTime
    xs <- ListT.toList . taker $ defaultCifar10set m
    t1 <- getCurrentTime
    printf "Loaded %s set of size %d in %s\n" desc (length xs) (show (t1 `diffUTCTime` t0))
    pure $ fmap (\(t, c) -> (t ^/ 255, c)) xs

   where
    taker = case ms of
              Just s -> ListT.take s
              Nothing -> id
    desc = case m of Train -> "training"; Test -> "testing"

epochs
  :: forall ch step . (ch ~ 3, step ~ 5)
  => [(Tensor '[3, 32, 32], Category)]    -- ^ test set
  -> HsReal                               -- ^ learning rate
  -> UTCTime                              -- ^ start time (for logging)
  -> Int                                  -- ^ number of epochs to run
  -> [(Tensor '[3, 32, 32], Category)]    -- ^ training set
  -> LeNet ch step                        -- ^ initial architecture
  -> IO ()
epochs test lr t0 mx tset net0 = do
  printf "initial "
  -- let
  --   f :: Tensor '[3,32,32]
  --   f = fst $ head test

  --   l :: [Tensor '[3, 32, 32]]
  --   l = map fst $ tail test
  -- print $ f Math.!! (dim :: Dim 0) Math.!! (dim :: Dim 0)
  -- print "=================="
  -- print "=================="
  -- print "=================="
  -- print "=================="
  -- print "=================="
  -- print "=================="
  -- print "=================="
  -- print $ (head l) Math.!! (dim :: Dim 0) Math.!! (dim :: Dim 0)

  testNet net0
  runEpoch 1 net0
  where
    runEpoch :: Int -> LeNet ch step -> IO ()
    runEpoch e net
      | e > mx    = pure ()
      | otherwise = do
        printf "\n[Epoch %d/%d]\n" e mx
        net' <- runBatches (dim :: Dim 4) lr t0 e tset net
        testNet net'
        runEpoch (e + 1) net'

    -- all input tensors for testing
    testX :: [Tensor '[3, 32, 32]]
    testX = map fst test

    -- all output categories for testing
    testY :: [Category]
    testY = map snd test

    testNet :: LeNet ch step -> IO ()
    testNet net = do
      -- mapM_
      --   (\(i,c) -> printf "truth: %s, inferred: %s\n" (show c) (show $ infer net i))
      --   test

      printf ("[test accuracy: %.1f%% / %d] All same? %s") (acc * 100 :: Float) (length testY)
        (if all (== head preds) preds then show (head preds) else "No.")
      hFlush stdout
     where
      preds = map (infer net) testX

      acc = genericLength (filter id $ zipWith (==) preds testY) / genericLength testY

runBatches
  :: forall ch step batch . (ch ~ 3, step ~ 5)
  => KnownNat (batch * 10)
  => KnownNat (batch)
  => KnownDim (batch * 10)
  => KnownDim (batch)
  => Dim batch
  -> HsReal
  -> UTCTime
  -> Int
  -> [(Tensor '[3, 32, 32], Category)]
  -> LeNet ch step
  -> IO (LeNet ch step)
runBatches d lr t00 e = go 0
 where
  go
    :: Int
    -> [(Tensor '[3, 32, 32], Category)]
    -> LeNet ch step
    -> IO (LeNet ch step)
  go !bid !tset !net = do
    let (batch, next) = splitAt (fromIntegral $ dimVal d) tset
    if null batch
    then pure net
    else do
      t0 <- getCurrentTime
      (net', hist) <- trainBatchStep lr (net, [])
        ( catArray0 (fmap (unsqueeze1d (dim :: Dim 0) . fst) batch) :: Tensor '[batch, 3, 32, 32]
        , fmap snd batch
        )
      t1 <- getCurrentTime

      --
      printf (setRewind ++ "(%d-batch #%d)[mse %.4f] in %s (total: %s)")
        (dimVal d) (bid+1)
        (P.sum . map ((`get1d` 0) . fst) $ hist)
        (show (t1 `diffUTCTime`  t0))
        (show (t1 `diffUTCTime` t00))
      hFlush stdout

      -- go again, using the next minibatch and new network.
      go (bid+1) next net'

 -- | Erase the last line in an ANSI terminal
clearLn :: IO ()
clearLn = printf "\ESC[2K"

-- | set rewind marker for 'clearLn'
setRewind :: String
setRewind = "\r"

-- | clear the screen in an ANSI terminal
clearScreen :: IO ()
clearScreen = putStr "\ESC[2J"

infer
  :: (ch ~ 3, step ~ 5)
  => LeNet ch step
  -> Tensor '[ch, 32, 32]
  -> Category
infer net

  -- cast from Integer to 'Torch.Data.Loaders.Cifar10.Category'
  = toEnum . fromIntegral

  -- Unbox the LongTensor '[1] to get 'Integer'
  . (`Long.get1d` 0)

  -- argmax the output Tensor '[10] distriubtion. Returns LongTensor '[1]
  . maxIndex1d

  . foo

 where
  foo x
    -- take an input tensor and run 'lenet' with the model (undefined is the
    -- learning rate, which we can ignore)
    = unsafePerformIO $ do
        -- print $ x Math.!! (dim :: Dim 0) Math.!! (dim :: Dim 0)
        let x' = evalBP2 (lenet undefined) net x
        -- print x'
        pure x'

trainStep
  :: (ch ~ 3, step ~ 5)
  => HsReal
  -> (LeNet ch step, [(Tensor '[1], Category)])
  -> (Tensor '[ch, 32, 32], Category)
  -> IO (LeNet ch step, [(Tensor '[1], Category)])
trainStep lr (net, hist) (x, y) = do
  pure (Bp.add net gnet, (out, y):hist)
  where
    (out, (gnet, _))
      = backprop2
        ( clip (-1000,1000)
        . mSECriterion (onehotT y)
        .: lenet lr)
        net x


trainBatchStep
  :: forall ch step batch
  .  (ch ~ 3, step ~ 5)
  => KnownDim batch
  => KnownNat batch
  => KnownDim (batch * 10)
  => KnownNat (batch * 10)
  => HsReal
  -> (LeNet ch step, [(Tensor '[1], [Category])])
  -> (Tensor '[batch, ch, 32, 32], [Category])
  -> IO (LeNet ch step, [(Tensor '[1], [Category])])
trainBatchStep lr (net, hist) (xs, ys) = do
  pure (Bp.add net gnet, (out,ys):hist)
  where
    out :: Tensor '[1]
    (out, (gnet, _))
      = backprop2
        ( mSECriterion ((unsafeMatrix $ fmap onehotf ys) :: Tensor '[batch, 10])
        .: lenetBatch lr)
        net xs


