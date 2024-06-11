{-# LANGUAGE CPP #-}
module Main where

import Test.Hspec (hspec)
import qualified BasicSpec
import qualified CudaSpec
import qualified GeneratorSpec
import qualified MemorySpec

main :: IO ()
main = hspec $ do
  BasicSpec.spec
#ifndef darwin_HOST_OS
  CudaSpec.spec
#endif
  GeneratorSpec.spec
  MemorySpec.spec
