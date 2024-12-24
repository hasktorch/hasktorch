{-# LANGUAGE CPP #-}
module Main where

import Test.Hspec (hspec)
import qualified BasicSpec
import qualified CudaSpec
import qualified MpsSpec
import qualified GeneratorSpec
import qualified MemorySpec

main :: IO ()
main = hspec $ do
  BasicSpec.spec
#ifdef darwin_HOST_OS
  MpsSpec.spec
#else
  CudaSpec.spec
#endif
  GeneratorSpec.spec
  MemorySpec.spec
