{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module PipelineSpec where

import Control.Concurrent
import Control.Concurrent.Async
import Control.Monad
import System.Exit
import System.IO
import System.Timeout
import Torch.Data.Pipeline
  
import Test.Hspec


defaultTimeout = 35000
timeoutConcurrent = 120000
shouldTimeout = 10000

newtype MockData = MockData Int
newtype FailureDataset = FailureDataset Int

-- this instance tests that after mzeros are returned a few times,
-- batches are still processed normally afterwards
instance Dataset IO  FailureDataset Int where
  getBatch (FailureDataset iters) _ = if iters < 5
    then mzero
    else threadDelay 10000 >> pure iters 
  numIters (FailureDataset iters)  = iters

instance Dataset IO MockData Int where
  getBatch (MockData iters) _ = threadDelay 10000 >> pure iters 
  numIters (MockData iters)  = iters

instance ConcurrentDataset IO MockData Int where
  getBatchConcurrently _ dataset iter = threadDelay 20000 >> pure iter

testFoldTimeout :: Dataset IO dataset Int => dataset -> IO ()
testFoldTimeout dataset = do
  (fold, thread ) <- makeFold' dataset
  timedOut <- async $ fold $ FoldM (takeBatchThenTimeout 10000) (pure 0) pure
  wait thread
  cancel timedOut

testConcurrentFoldTimeout :: MockData  -> Int -> IO ()
testConcurrentFoldTimeout dataset numWorkers = do
  (fold, threads) <- makeConcurrentFold' id dataset numWorkers 
  timedOut <- async $ fold $  FoldM (takeBatchThenTimeout 10000) (pure 0) pure
  mapM_ wait threads
  cancel timedOut

takeBatchThenTimeout :: Int -> Int -> Int -> IO Int
takeBatchThenTimeout timeout _ input =  threadDelay timeout >> pure input

-- | This function returns Nothing if the IO action takes longer than the given
-- | time, otherwise it returns Just()
runTest :: Int -> IO () -> IO (Maybe ())
runTest time test = do
    hFlush stdout
    result <- timeout time test
    pure result

-- | Returns just if the given test times out
runTestExpectTimeout :: Int -> IO () -> IO (Maybe ())
runTestExpectTimeout time test = do
    result <- timeout defaultTimeout test
    case result of
        Nothing -> pure (Just ())
        Just _  -> pure Nothing

-- | The first test tests if batches are being streamed ahead of
-- | the fold function for consumption. A new batch should be processed as soon as 
-- | the fold consumes a batch.
-- | 
-- | The second test tests that with 2 workers and fold processing batches twice as fast
-- | as they are yielded that workers are never idling. If they are the test must fail.
-- | Diagrammatically, this is how things should work out: 
-- | 
-- | working = -, idle = .
-- | Worker 1: |-----|-----|-----|
-- | Worker 2: |-----|-----|-----|
-- | Fold:     |.....|--|--|--|--|
spec = do
  it "Tests data is flowing" $
    (runTest defaultTimeout (testFoldTimeout $ MockData 2)) `shouldReturn` (Just ())
  it "Tests concurrent datasets yield concurrently" $
    (runTest timeoutConcurrent (testConcurrentFoldTimeout (MockData 3) 2)) `shouldReturn` (Just ())
  it "Tests that failures in dataset processing aren't yielded" $
    (runTest 10000 (testFoldTimeout $ FailureDataset 4)) `shouldReturn` (Just ())
  it "Tests batches are still yielded after failures" $
    (runTestExpectTimeout 10000 (testFoldTimeout $ FailureDataset 5)) `shouldReturn` (Just ())
