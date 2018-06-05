module Train where

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


import Torch.Double as Math hiding (Sum)
import qualified Torch.Long as Long

import LeNet
import DataLoader

main :: IO ()
main = do
  t0 <- liftIO getCurrentTime
  trainset <- ListT.toList . ListT.take (10) $ cifar10set Train
  t1 <- liftIO getCurrentTime
  printf "Loaded training set in %s\n" (show (t1 `diffUTCTime` t0))
  testset  <- ListT.toList $ cifar10set Test

  net0 <- newLeNet
  -- epochs 2 trainset net0
  lastnet <- runBatches 1 trainset net0
  print lastnet

accuracy :: [(Tensor '[10], Integer)] -> Double
accuracy xs = foldl go 0 xs / genericLength xs
  where
    go :: Double -> (Tensor '[10], Integer) -> Double
    go acc (pred, y) = acc + fromIntegral (fromEnum (y == fromIntegral (Long.get1d (maxIndex1d pred) 0)))

epochs :: Int -> [(Tensor '[3, 32, 32], Integer)] -> LeNet -> IO ()
epochs mx tset = runEpoch 1
  where
    runEpoch :: Int -> LeNet -> IO ()
    runEpoch e net
      | e > mx    = putStrLn "Done!"
      | otherwise = do
        print (e, "e")
        printf "[Epoch %d]\n" e
        net' <- runBatches 1 tset net
        runEpoch (e + 1) net'

runBatches :: Int -> [(Tensor '[3, 32, 32], Integer)] -> LeNet -> IO LeNet
runBatches b trainset net = do
  if null trainset
  then pure net
  else do
    print "x"
    let (batch, next) = splitAt 10 trainset
    net' <- go b batch net
    runBatches (b+1) next net'
  where
    go
      :: Int
      -> [(Tensor '[3, 32, 32], Integer)]
      -> LeNet
      -> IO LeNet
    go b chunk net0 = do
      printf "(Batch %d) Training on %d points\t" b (length chunk)
      t0 <- liftIO getCurrentTime
      (net', hist) <- foldM step (net0, []) chunk
      let
        acc = accuracy hist
      printf "[Train accuracy: %.4f]\t" acc
      t1 <- liftIO getCurrentTime
      printf "in %s\n" (show (t1 `diffUTCTime` t0))
      pure net'

step
  :: (LeNet, [(Tensor '[10], Integer)])
  -> (Tensor '[3, 32, 32], Integer)
  -> IO (LeNet, [(Tensor '[10], Integer)])
step (net, hist) (x, y) = do
  let x = (net', (out, y):hist)
  pure x
  where
    (out, (net', _)) = backprop2 (lenet 0.1) net x


