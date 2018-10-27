{-# LANGUAGE CPP #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Criterion.Main
import Criterion.Types (resamples)
import GHC.Conc (numCapabilities)
-- import Data.Time
-- import Text.Printf
import System.Random.Shuffle (shuffleM)
import qualified ListT
import Control.Monad.Trans.Class
import ImageLoading (img_loading_bench)

#ifdef CUDA
import Torch.Cuda.Double
#else
import Torch.Double
#endif


main :: IO ()
main = defaultMainWith (defaultConfig {resamples=5})
  [ bgroup "ImgLoading" img_loading_bench
  ]


