{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module PipelineSpec where

import System.IO
import System.Timeout
import System.Exit
import Control.Monad
import Control.Concurrent
import Control.Concurrent.Async
import Pipes
import Pipes.Concurrent
import Torch.Data.Pipeline
  
import Test.Hspec


defaultTimeout :: Int
defaultTimeout = 100000  -- 1 second

newtype MockData = MockData Int
instance Dataset MockData Int where
  getBatch (MockData iters) _ = pure $ iters
  numIters (MockData iters)  = iters

instance ConcurrentDataset MockData Int where
  getBatchConcurrently _ = getBatch 

testFoldTimeout :: MockData -> IO ()
testFoldTimeout dataset = do
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
  thread <- async $ runEffect $ readBatches dataset toBatches
  fold <- pure $ foldFromProducer (takeBatch fromBatches)
  timedOut <- async $ fold $ FoldM takeBatchThenTimeout (pure 0) pure
  wait thread >> cancel timedOut
  pure ()


testConcurrentFoldTimeout :: MockData  -> Int -> IO ()
testConcurrentFoldTimeout dataset numWorkers = do
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded numWorkers)
  threads <- forM [1..numWorkers] $ \workerId -> async $ runEffect $ readBatchesConcurrently workerId dataset toBatches
  fold <- pure $ foldFromProducer (takeBatch fromBatches)
  timedOut <- async $ fold $  FoldM takeBatchThenTimeout (pure 0) pure

  mapM_ wait threads >> cancel timedOut
  pure ()

takeBatchThenTimeout :: Int -> Int -> IO Int
takeBatchThenTimeout _ input = threadDelay 50000 >> pure input

runTest :: IO () -> IO (Maybe ())
runTest test = do
    hFlush stdout
    result <- timeout defaultTimeout test
    pure result

spec = do
  it "Tests data is flowing" $
    (runTest (testFoldTimeout $ MockData 2)) `shouldReturn` (Just ())
  it "Tests concurrent datasets yield concurrently" $
    (runTest (testConcurrentFoldTimeout (MockData 1) 4) "yo") `shouldReturn` (Just ())
