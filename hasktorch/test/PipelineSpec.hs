{-# LANGUAGE CPP #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module PipelineSpec where

import Control.Concurrent
import Control.Concurrent.Async
import Control.Monad
import Control.Monad.Cont (ContT (runContT))
import GHC.Exts (IsList (fromList))
import GHC.IO (unsafePerformIO)
import Pipes
import Pipes.Prelude (drain)
import qualified Pipes.Prelude as P
import System.Exit
import System.IO
import System.Random
import System.Timeout
import Test.Hspec
import Torch.Data.Pipeline
import Torch.Data.Utils

streamAheadTimeout = 25000

timeoutConcurrent = 85000

data MockData = MockData

data ConcurrentData = ConcurrentData

data ShuffleSet = ShuffleSet

-- | Yields 4 items, each item taking 10000 milliseconds to compute
instance Dataset IO ConcurrentData Int Int where
  getItem ConcurrentData k = threadDelay 10000 >> pure k
  keys ConcurrentData = fromList [0 .. 7]

-- | Yields 2 items, each taking 5000 milliseconds to compute
instance Dataset IO MockData Int Int where
  getItem MockData _ = threadDelay 10000 >> pure 0
  keys (MockData) = fromList [0 .. 1]

instance Dataset IO ShuffleSet Int Int where
  getItem ShuffleSet k = pure k
  keys _ = fromList [0 .. 100]

testFoldTimeout :: MockData -> IO ()
testFoldTimeout dataset = do
  runContT (streamFromMap (datasetOpts 1) dataset) $
    (\l -> runEffect $ enumerateData l >-> takeThenTimeout) . fst
  where
    takeThenTimeout = forever $ do
      (_, iter) <- await
      lift $ when (iter == 0) $ threadDelay 5000

testConcurrentFoldTimeout :: ConcurrentData -> Int -> IO ()
testConcurrentFoldTimeout dataset numWorkers = do
  runContT (streamFromMap (datasetOpts numWorkers) dataset) $
    (\l -> runEffect $ enumerateData l >-> takeThenTimeout) . fst
  where
    takeThenTimeout = forever $ do
      (_, iter) <- await
      -- don't timeout on the last two iterations since data shouldn't be
      -- getting yielded anymore
      lift $ when (iter < 7) $ threadDelay 5000

testShuffle :: IO ()
testShuffle = do
  let options = (datasetOpts 4) {shuffle = Shuffle $ mkStdGen 123}
  let optionsDiff = (datasetOpts 4) {shuffle = Shuffle $ mkStdGen 50}

  datasets <- replicateM 100 $ runContT (streamFromMap options ShuffleSet) $ P.toListM . enumerate . fst
  differentOrder <- runContT (streamFromMap optionsDiff ShuffleSet) $ P.toListM . enumerate . fst

  all (\elems -> elems == head datasets) datasets `shouldBe` True
  head datasets == differentOrder `shouldBe` False

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
  it "Test data is flowing" $
    (runTest streamAheadTimeout (testFoldTimeout MockData)) `shouldReturn` (Just ())
  it "Test concurrent datasets yield concurrently" $
    (runTest timeoutConcurrent (testConcurrentFoldTimeout ConcurrentData 2) `shouldReturn` (Just ()))
#endif
  it "Test shuffle is deterministic with seed" $ testShuffle
