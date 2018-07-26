{-# LANGUAGE TypeFamilies #-}
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

main :: IO ()
main = do
  print (constant 0 :: Tensor '[3,2,2])
  runExceptT (im2torch "/mnt/lake/datasets/cifar-10/train/truck/48151_truck.png") >>= print
  runExceptT (im2torch "/mnt/lake/datasets/cifar-10/train/truck/48151_truck.png") >>= print
  -- runExceptT (im2torch "/mnt/lake/datasets/cifar-10/train/truck/48151_truck.png") >>= print
  -- runExceptT (im2torch "/mnt/lake/datasets/cifar-10/train/truck/48151_truck.png") >>= print
  t0 <- liftIO getCurrentTime
  trainset <- ListT.toList {- . ListT.take (5) -} $ defaultCifar10set Train
  t1 <- liftIO getCurrentTime
  -- print $ ListT.length trainset
  printf "Loaded training set in %s\n" (show (t1 `diffUTCTime` t0))
  -- testset <- ListT.toList $ defaultCifar10set Test
  print $ length trainset
  -- Cuda.get >>= Cuda.props >>= print

  -- print "matrix"
  -- let Right (t :: Tensor '[2,3]) = matrix [[2,3,4], [3,4,5]]
  -- print t

  -- net0 <- newLeNet @3 @5

  -- t :: Tensor '[1,2,3] <- uniform (-1) 1
  -- print t
  -- print "starting"
  -- print net0
  -- epochs 1 trainset net0
  -- lastnet <- runBatches 1 trainset net0
  -- print lastnet
  print "done!"

--epochs
--  :: forall ch step . (ch ~ 3, step ~ 5)
--  => Int -> [(Tensor '[3, 32, 32], Integer)] -> LeNet ch step -> IO ()
--epochs mx tset = runEpoch 1
--  where
--    runEpoch :: Int -> LeNet ch step -> IO ()
--    runEpoch e net
--      | e > mx    = putStrLn "Done!"
--      | otherwise = do
--        print (e, "e")
--        printf "[Epoch %d]\n" e
--        net' <- runBatches 1 tset net
--        runEpoch (e + 1) net'
--
--runBatches
--  :: forall ch step . (ch ~ 3, step ~ 5)
--  => Int
--  -> [(Tensor '[3, 32, 32], Integer)]
--  -> LeNet ch step
--  -> IO (LeNet ch step)
--runBatches b trainset net = do
--  if null trainset
--  then pure net
--  else do
--    print "x"
--    let (batch, next) = splitAt 10 trainset
--    net' <- go b batch net
--    runBatches (b+1) next net'
--  where
--    go
--      :: Int
--      -> [(Tensor '[3, 32, 32], Integer)]
--      -> LeNet ch step
--      -> IO (LeNet ch step)
--    go b chunk net0 = do
--      printf "(Batch %d) Training on %d points\t" b (length chunk)
--      t0 <- liftIO getCurrentTime
--      (net', hist) <- foldM step (net0, []) chunk
--      let
--        acc = accuracy hist
--      printf "[Train accuracy: %.4f]\t" acc
--      t1 <- liftIO getCurrentTime
--      printf "in %s\n" (show (t1 `diffUTCTime` t0))
--      pure net'
--
--step
--  :: (ch ~ 3, step ~ 5)
--  => (LeNet ch step, [(Tensor '[10], Integer)])
--  -> (Tensor '[ch, 32, 32], Integer)
--  -> IO (LeNet ch step, [(Tensor '[10], Integer)])
--step (net, hist) (x, y) = do
--  let x = (net', (out, y):hist)
--  pure x
--  where
--    (out, (net', _)) = backprop2 (lenet 0.1) net x
--
--
