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
import Numeric.Backprop
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
  xs  <- loadTrain 500
  net0 <- newLeNet @3 @5
  print net0
  putStrLn "Start training:"
  t0 <- getCurrentTime
  net <- epochs 0.01 t0 3 xs net0
  t1 <- getCurrentTime
  printf "\nFinished training\n"
  -- xs' <- loadTest  50
  -- lastnet <- runBatches 1 trainset net0
  -- print lastnet

  putStrLn "\ndone!"
 where
  loadTest  = loadData Test
  loadTrain = loadData Train
  loadData m s = do
    t0 <- getCurrentTime
    xs <- ListT.toList . ListT.take s $ defaultCifar10set m
    t1 <- getCurrentTime
    let desc = case m of Train -> "training"; Test -> "testing"
    printf "Loaded %s set of size %d in %s\n" desc (length xs) (show (t1 `diffUTCTime` t0))
    pure xs

epochs
  :: forall ch step . (ch ~ 3, step ~ 5)
  => HsReal
  -> UTCTime
  -> Int
  -> [(Tensor '[3, 32, 32], Integer)]
  -> LeNet ch step
  -> IO ()
epochs lr t0 mx tset = runEpoch 1
  where
    runEpoch :: Int -> LeNet ch step -> IO ()
    runEpoch e net
      | e > mx    = pure ()
      | otherwise = do
        net' <- runBatches lr t0 e 10 tset net
        runEpoch (e + 1) net'

runBatches
  :: forall ch step . (ch ~ 3, step ~ 5)
  => HsReal
  -> UTCTime
  -> Int
  -> Int
  -> [(Tensor '[3, 32, 32], Category)]
  -> LeNet ch step
  -> IO (LeNet ch step)
runBatches lr t00 e bsize = go 0
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
      (net', hist) <- foldM (step lr) (net, []) batch
      t1 <- getCurrentTime
      printf (setRewind ++ "[Epoch %d](%d-batch #%d)[accuracy: %.4f] in %s (total: %s)")
        e bsize (bid+1)
        (accuracy hist)
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

step
  :: (ch ~ 3, step ~ 5)
  => HsReal -> (LeNet ch step, [(Tensor '[10], Category)])
  -> (Tensor '[ch, 32, 32], Category)
  -> IO (LeNet ch step, [(Tensor '[10], Category)])
step lr (net, hist) (x, y) = pure (net', (out, y):hist)
  where
    (out, (net', _)) = backprop2 (lenet lr) net x

trainStep
  :: (ch ~ 3, step ~ 5)
  => HsReal -> (LeNet ch step, [(Tensor '[10], Category)])
  -> (Tensor '[ch, 32, 32], Category)
  -> IO (LeNet ch step, [(Tensor '[10], Category)])
trainStep lr (net, hist) (x, y) = pure (net', (out, y):hist)
  where
    (out, (net', _)) = backprop2 (classNLLCriterion (onehot y) .: lenet lr) net x

-- (classNLLCriterion (Long.unsafeVector [0,1]))

