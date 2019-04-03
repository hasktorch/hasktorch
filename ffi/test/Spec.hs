{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

--import Control.Exception.Safe (throwString, throw)
import Test.Hspec
import Aten.Const
import Aten.Type
import Aten.Type.TensorOptions
import Aten.Type.Tensor
import Aten.Native


main :: IO ()
main = hspec $ do
  describe "Basic Tensor Spec" $ do
    describe "Create Tensor Spec" createSpec


createSpec :: Spec
createSpec = do
  it "Create Tensor" $ do
    to <- newTensorOptions kCPU
    tod <- tensorOptions_dtype to kByte
    return ()
