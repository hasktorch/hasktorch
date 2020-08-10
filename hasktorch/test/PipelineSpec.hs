{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE CPP #-}
module PipelineSpec where

import Control.Concurrent
import Control.Concurrent.Async
import Control.Monad
import System.Exit
import System.IO
import System.Timeout
import Torch.Data.Pipeline
import Pipes

import Test.Hspec
import GHC.IO (unsafePerformIO)
import Control.Monad.Cont (ContT(runContT))
import Pipes.Prelude (drain)


streamAheadTimeout = 20000
timeoutConcurrent = 100000

data MockData = MockData 
data ConcurrentData = ConcurrentData 

-- | Yields 4 items, each item taking 10000 milliseconds to compute
instance Dataset ConcurrentData Int where
  getItem ConcurrentData  _ = unsafePerformIO $ threadDelay 10000 >> pure 0
  size ConcurrentData = 8 
  
-- | Yields 2 items, each taking 5000 milliseconds to compute
instance Dataset MockData Int where
  getItem (MockData ) _ = unsafePerformIO $ threadDelay 5000 >> pure 0
  size (MockData) = 2


testFoldTimeout :: MockData -> IO ()
testFoldTimeout dataset = do
   runContT (makeListT (mapStyleOpts 1) dataset sequentialSampler id) $
     (\l -> runEffect $ enumerate l >-> takeThenTimeout)
  where takeThenTimeout = forever $ do
          (_, iter) <- await
          lift $ when (iter == 0 ) $ threadDelay 10000
          
          
          

testConcurrentFoldTimeout :: ConcurrentData -> Int -> IO ()
testConcurrentFoldTimeout dataset numWorkers = do
   runContT (makeListT (mapStyleOpts numWorkers) dataset sequentialSampler id) $
     (\l -> runEffect $ enumerate l >-> takeThenTimeout)
  where takeThenTimeout = forever $ do
          (_, iter) <- await
          -- don't timeout on the last two iterations since data shouldn't be
          -- getting yielded anymore
          lift $ when (iter < 7) $ threadDelay 5000


-- | This function returns Nothing if the IO action takes longer than the given
-- | time, otherwise it returns Just ()
runTest :: Int -> IO () -> IO (Maybe ())
runTest time test = do
    hFlush stdout
    result <- timeout time test
    pure result


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
-- | Fold:     |.....|--|--|--|--|--|--|
spec :: Spec
spec = do
#ifndef darwin_HOST_OS
  it "Tests data is flowing" $
    (runTest streamAheadTimeout (testFoldTimeout MockData)) `shouldReturn` (Just ())
  it "Tests concurrent datasets yield concurrently" $
    (runTest timeoutConcurrent (testConcurrentFoldTimeout ConcurrentData 2) `shouldReturn` (Just ()))
#endif
  return () -- no empty do block on macos
