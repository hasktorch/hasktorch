{-# LANGUAGE CPP #-}
module Main where

import Criterion.Main
import Criterion.Types (resamples)
import GHC.Conc (numCapabilities)
-- import Data.Time
-- import Text.Printf
import System.Random.Shuffle (shuffleM)
import qualified ListT
import Control.Monad.Trans.Class

import Torch.Data.Loaders.Cifar10

#ifdef CUDA_
import Torch.Cuda.Double
#else
import Torch.Double
#endif


main :: IO ()
main = defaultMainWith (defaultConfig {resamples=5}) [
  bgroup "cifar10loader"
    [ bench ( "1-threadmax, numCapabilities: " ++ show numCapabilities) $ nfIO (go 1)
    -- , bench ( "5-threadmax, numCapabilities: " ++ show numCapabilities) $ nfIO (go 5)
    -- , bench ("10-threadmax, numCapabilities: " ++ show numCapabilities) $ nfIO (go 10)
    -- , bench ("15-threadmax, numCapabilities: " ++ show numCapabilities) $ nfIO (go 15)
    , bench ("20-threadmax, numCapabilities: " ++ show numCapabilities) $ nfIO (go 20)
    ]
  ]
 where
  go mx = ListT.toList $ do
    (t, c) <- cifar10set mx default_cifar_path Test
    xs <- lift $ tensordata t
    pure (xs, c)
  -- loadData m ms = do
  --   t0 <- getCurrentTime
  --   xs <- ListT.toList . taker $ defaultCifar10set m
  --   t1 <- getCurrentTime
  --   printf "Loaded %s set of size %d in %s\n" desc (length xs) (show (t1 `diffUTCTime` t0))
  --   shuffleM xs

  --  where
  --   taker =
  --     case ms of
  --       Just s -> ListT.take s
  --       Nothing -> id
  --   desc =
  --     case m of
  --       Train -> "training"
  --       Test -> "testing"


