{-# LANGUAGE ForeignFunctionInterface #-}

module TestRawInterface (testRawInterface) where

import THDoubleTensor as TR
import THDoubleTensorMath as TR
import THTypes
import TorchStructs

import Foreign.C.Types
import Test.Hspec

testRawInterface = do
  putStrLn "Raw Types"

  putStrLn "Done"
