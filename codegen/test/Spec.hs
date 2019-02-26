{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Control.Exception.Safe (throwString, throw)
import Data.Proxy
import ParseNativeFunctions (NativeFunction, NativeFunction')
import ParseNN (NN, NN')
import ParseDerivatives (Derivative)
import System.Directory (doesFileExist)
import Test.Hspec
import qualified Data.Yaml as Y
import qualified ParseFunctionSig as F
import qualified ParseNativeFunctions as NF
import qualified RenderNativeFunctions as RNF

main :: IO ()
main = hspec $ do
  describe "parsing native_functions.yaml" $ do
    describe "NativeFunction Spec" nativeFunctionsSpec
  describe "parsing nn.yaml" $ do
    describe "NN Spec" nnSpec
  describe "parsing derivatives.yaml" $ do
    describe "Derivatives Spec" nnSpec

nativeFunctionsPath :: FilePath
nativeFunctionsPath = "../spec/native_functions_modified.yaml"

nativeFunctionsSpec :: Spec
nativeFunctionsSpec = do
  xs <- runIO $ vanillaParse nativeFunctionsPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ NativeFunction)
    (length fs) `shouldBe` (length xs)

  it "parses the same number of typed functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ NativeFunction')
    (length fs) `shouldBe` (length xs)

  it "parses the `_cudnn_ctc_loss` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ filter (("_cudnn_ctc_loss" ==) . F.name . NF.func') fs of
      Nothing -> fail "_cudnn_ctc_loss function not found!"
      Just nf -> do
        F.parameters nf `shouldBe`
          [ F.Parameter (F.TenType F.Tensor) "log_probs" Nothing
          , F.Parameter (F.TenType F.Tensor) "targets" Nothing
          , F.Parameter (F.TenType (F.IntList (Just []))) "input_lengths" Nothing
          , F.Parameter (F.TenType (F.IntList (Just []))) "target_lengths" Nothing
          , F.Parameter (F.CType F.CInt)  "blank" Nothing
          , F.Parameter (F.CType F.CBool) "deterministic" Nothing
          , F.Parameter (F.CType F.CBool) "zero_infinity" Nothing
          ]
        F.retType nf `shouldBe` F.Tuple [F.TenType F.Tensor, F.TenType F.Tensor]
  it "parses the `add_out` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ tail $ filter (("add" ==) . F.name . NF.func') fs of
      Nothing -> fail "add_out function not found!"
      Just nf -> do
        F.parameters nf `shouldBe`
          [ F.Parameter (F.TenType F.Tensor) "self" Nothing
          , F.Parameter (F.TenType F.Tensor) "other" Nothing
          , F.Star
          , F.Parameter (F.TenType F.Scalar) "alpha" (Just (F.ValInt 1))
          , F.Parameter (F.TenType F.TensorA') "out" Nothing
          ]
        F.retType nf `shouldBe` F.TenType F.TensorA'
  it "parses the `add_out` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ tail $ filter (("add" ==) . F.name . NF.func') fs of
      Nothing -> fail "add_out function not found!"
      Just nf -> do
        let nf' = RNF.removeStarArgument (RNF.Common, nf)
        F.parameters nf' `shouldBe`
          [ F.Parameter (F.TenType F.TensorA') "out" Nothing
          , F.Parameter (F.TenType F.Tensor) "self" Nothing
          , F.Parameter (F.TenType F.Tensor) "other" Nothing
          , F.Parameter (F.TenType F.Scalar) "alpha" (Just (F.ValInt 1))
          ]
        F.retType nf' `shouldBe` F.TenType F.TensorA'
  it "parses the `thnn_conv_transpose2d_backward` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ filter (("thnn_conv_transpose2d_backward" ==) . F.name . NF.func') fs of
      Nothing -> fail "thnn_conv_transpose2d_backward function not found!"
      Just nf -> do
        F.parameters nf `shouldBe`
          [
            F.Parameter (F.TenType F.Tensor) "grad_output" (Nothing)
          , F.Parameter (F.TenType F.Tensor) "self" (Nothing)
          , F.Parameter (F.TenType F.Tensor) "weight" (Nothing)
          , F.Parameter (F.TenType (F.IntList (Just [2]))) "kernel_size" (Nothing)
          , F.Parameter (F.TenType (F.IntList (Just [2]))) "stride" (Nothing)
          , F.Parameter (F.TenType (F.IntList (Just [2]))) "padding" (Nothing)
          , F.Parameter (F.TenType (F.IntList (Just [2]))) "output_padding" (Nothing)
          , F.Parameter (F.TenType (F.IntList (Just [2]))) "dilation" (Nothing)
          , F.Parameter (F.TenType F.Tensor) "columns" (Nothing)
          , F.Parameter (F.TenType F.Tensor) "ones" (Nothing)
          , F.Star
          , F.Parameter (F.TenType F.TensorAQ') "grad_input" (Nothing)
          , F.Parameter (F.TenType F.TensorAQ') "grad_weight" (Nothing)
          , F.Parameter (F.TenType F.TensorAQ') "grad_bias" (Nothing)
          ]
        F.retType nf `shouldBe` F.Tuple [F.TenType F.TensorA',F.TenType F.TensorA',F.TenType F.TensorA']


 where
  mhead :: [a] -> Maybe a
  mhead = \case
    [] -> Nothing
    a:as -> Just a

  parseWith :: forall x funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither nativeFunctionsPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

nnPath :: FilePath
nnPath = "../spec/nn.yaml"

nnSpec :: Spec
nnSpec = do
  xs <- runIO $ vanillaParse nnPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ NN)
    (length fs) `shouldBe` (length xs)

  it "parses the same number of typed functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ NN')
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall x funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither nnPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

derivativesPath :: FilePath
derivativesPath = "../spec/derivatives.yaml"

derivativesSpec :: Spec
derivativesSpec = do
  xs <- runIO $ vanillaParse derivativesPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ Derivative)
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall x funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither derivativesPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString "Spec doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp

