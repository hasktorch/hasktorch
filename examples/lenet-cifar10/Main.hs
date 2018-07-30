{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
module Main where

import Data.Typeable
import Data.List
import Control.Monad
import Control.Monad.Loops
import Data.Monoid
import Data.Time
import Control.Monad.IO.Class
import Prelude as P
import Text.Printf
import ListT (ListT)
import qualified ListT
import Numeric.Backprop as Bp
import Numeric.Dimensions
import System.IO.Unsafe
import Control.Concurrent
-- import qualified Foreign.CUDA as Cuda

#ifdef CUDA
import Torch.Cuda.Double as Math hiding (Sum)
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double as Math hiding (Sum)
import qualified Torch.Long as Long
#endif

import Torch.Models.LeNet
import Torch.Data.Loaders.Cifar10
import Torch.Data.Metrics

import Control.Monad.Trans.Except
import System.IO (hFlush, stdout)

main :: IO ()
main = do
  clearScreen
  xs  <- loadTrain Nothing -- $ Just 50
  xs' <- loadTest  Nothing -- $ Just 10
  net0 <- newLeNet @3 @5
  print net0
  putStrLn "Start training:"
  t0 <- getCurrentTime
  net <- epochs xs' 0.0001 t0 1 xs net0
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
    pure xs

   where
    taker = case ms of
              Just s -> ListT.take s
              Nothing -> id
    desc = case m of Train -> "training"; Test -> "testing"

epochs
  :: forall ch step . (ch ~ 3, step ~ 5)
  => [(Tensor '[3, 32, 32], Category)]
  -> HsReal
  -> UTCTime
  -> Int
  -> [(Tensor '[3, 32, 32], Category)]
  -> LeNet ch step
  -> IO ()
epochs test lr t0 mx tset net0 = do
  printf "initial "
  testNet net0
  runEpoch 1 net0
  where
    runEpoch :: Int -> LeNet ch step -> IO ()
    runEpoch e net
      | e > mx    = pure ()
      | otherwise = do
        net' <- run1Batches lr t0 e 50 tset net
        testNet net'
        runEpoch (e + 1) net'

    testX = map fst test
    testY = map snd test

    testNet :: LeNet ch step -> IO ()
    testNet net = do
      printf ("[test accuracy: %.1f%% / %d]") (acc * 100 :: Float) (length testY)
      hFlush stdout
     where
      preds = map (infer net) testX
      acc = genericLength (filter id $ zipWith (==) preds testY) / genericLength testY

run1Batches
  :: forall ch step . (ch ~ 3, step ~ 5)
  => HsReal
  -> UTCTime
  -> Int
  -> Int
  -> [(Tensor '[3, 32, 32], Category)]
  -> LeNet ch step
  -> IO (LeNet ch step)
run1Batches lr t00 e bsize = go 0
 where
  go
    :: Int
    -> [(Tensor '[3, 32, 32], Category)]
    -> LeNet ch step
    -> IO (LeNet ch step)
  go !bid !tset !net = do
    let (batch, next) = splitAt bsize tset
    if null batch
    then pure net
    else do
      t0 <- getCurrentTime
      (net', hist) <- foldM (trainStep lr) (net, []) batch
      t1 <- getCurrentTime
      printf (setRewind ++ "[Epoch %d](%d-batch #%d)[nlloss: %.4f] in %s (total: %s)")
        e bsize (bid+1)
        (P.sum . map ((`get1d` 0) . fst) $ hist)
        (show (t1 `diffUTCTime`  t0))
        (show (t1 `diffUTCTime` t00))
      hFlush stdout
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
  = toEnum
  . fromIntegral
  . (`Long.get1d` 0)
  . maxIndex1d
  . evalBP2 (lenet undefined) net

trainStep
  :: (ch ~ 3, step ~ 5)
  => HsReal
  -> (LeNet ch step, [(Tensor '[1], Category)])
  -> (Tensor '[ch, 32, 32], Category)
  -> IO (LeNet ch step, [(Tensor '[1], Category)])
trainStep lr (net, hist) (x, y) = do
  print out
  pure (Bp.add net gnet, (out, y):hist)
  where
    (out, (gnet, _))
      = backprop2
        ( classNLLCriterion (esingleton y)
        . unsqueeze1dBP (dim :: Dim 0)
        .: lenet lr)
        net x


