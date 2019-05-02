{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

--import Control.Exception.Safe (throwString, throw)
import Test.Hspec
import Aten.Const
import Aten.Type
import Aten.Managed.Type.TensorOptions
import Aten.Managed.Type.Tensor
import Aten.Managed.Type.IntArray
import Aten.Managed.Native


main :: IO ()
main = hspec $ do
  describe "Basic Tensor Spec" $ do
    describe "Create Tensor Spec" createSpec


createSpec :: Spec
createSpec = do
  it "Create Tensor" $ do
    to <- newTensorOptions_D kCPU
    dsize <- newIntArray
    intArray_push_back_l dsize 2
    intArray_push_back_l dsize 2
    tod <- tensorOptions_dtype_s to kByte
    ten <- zeros_lo dsize tod
    tensor_print ten
    return ()
