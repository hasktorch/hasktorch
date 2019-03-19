{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Control.Exception.Safe (throwString, throw)
import Data.Proxy
import Text.Megaparsec (parse, errorBundlePretty)
import ParseNativeFunctions (NativeFunction, NativeFunction')
import ParseNN (NN, NN')
import ParseDerivatives (Derivative)
import System.Directory (doesFileExist)
import Test.Hspec
import qualified Data.Yaml as Y
import ParseFunctionSig
import qualified ParseNativeFunctions as NF
import qualified ParseNN as NN
import qualified ParseHeadersForNN as HNN
import qualified RenderNativeFunctions as RNF
import qualified RenderNN as RNN

main :: IO ()
main = hspec $ do
  describe "parsing native_functions.yaml" $ do
    describe "NativeFunction Spec" nativeFunctionsSpec
--  describe "parsing nn.yaml" $ do
--    describe "NN Spec" nnSpec
--  describe "parsing THNN.h and THCUNN.h" $ do
--    describe "THNN.h Spec" thnnSpec
  describe "parsing derivatives.yaml" $ do
    describe "Derivatives Spec" derivativesSpec
  describe "parsing Declarations.yaml" $ do
    describe "Declarations Spec" declarationsSpec

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

    case fmap NF.func' $ mhead $ filter (("_cudnn_ctc_loss" ==) . name . NF.func') fs of
      Nothing -> fail "_cudnn_ctc_loss function not found!"
      Just nf -> do
        parameters nf `shouldBe`
          [ Parameter (TenType Tensor) "log_probs" Nothing
          , Parameter (TenType Tensor) "targets" Nothing
          , Parameter (TenType (IntList (Just []))) "input_lengths" Nothing
          , Parameter (TenType (IntList (Just []))) "target_lengths" Nothing
          , Parameter (CType CInt)  "blank" Nothing
          , Parameter (CType CBool) "deterministic" Nothing
          , Parameter (CType CBool) "zero_infinity" Nothing
          ]
        retType nf `shouldBe` Tuple [TenType Tensor, TenType Tensor]
  it "parses the `add_out` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ tail $ filter (("add" ==) . name . NF.func') fs of
      Nothing -> fail "add_out function not found!"
      Just nf -> do
        parameters nf `shouldBe`
          [ Parameter (TenType Tensor) "self" Nothing
          , Parameter (TenType Tensor) "other" Nothing
          , Star
          , Parameter (TenType Scalar) "alpha" (Just (ValInt 1))
          , Parameter (TenType TensorA') "out" Nothing
          ]
        retType nf `shouldBe` TenType TensorA'
  it "parses the `add_out` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ tail $ filter (("add" ==) . name . NF.func') fs of
      Nothing -> fail "add_out function not found!"
      Just nf -> do
        let nf' = RNF.removeStarArgument (RNF.Common, nf)
        parameters nf' `shouldBe`
          [ Parameter (TenType TensorA') "out" Nothing
          , Parameter (TenType Tensor) "self" Nothing
          , Parameter (TenType Tensor) "other" Nothing
          , Parameter (TenType Scalar) "alpha" (Just (ValInt 1))
          ]
        retType nf' `shouldBe` TenType TensorA'
  it "parses the `thnn_conv_transpose2d_backward` function" $ do
    fs <- parseWith (Proxy @ NativeFunction')

    case fmap NF.func' $ mhead $ filter (("thnn_conv_transpose2d_backward" ==) . name . NF.func') fs of
      Nothing -> fail "thnn_conv_transpose2d_backward function not found!"
      Just nf -> do
        parameters nf `shouldBe`
          [
            Parameter (TenType Tensor) "grad_output" (Nothing)
          , Parameter (TenType Tensor) "self" (Nothing)
          , Parameter (TenType Tensor) "weight" (Nothing)
          , Parameter (TenType (IntList (Just [2]))) "kernel_size" (Nothing)
          , Parameter (TenType (IntList (Just [2]))) "stride" (Nothing)
          , Parameter (TenType (IntList (Just [2]))) "padding" (Nothing)
          , Parameter (TenType (IntList (Just [2]))) "output_padding" (Nothing)
          , Parameter (TenType (IntList (Just [2]))) "dilation" (Nothing)
          , Parameter (TenType Tensor) "columns" (Nothing)
          , Parameter (TenType Tensor) "ones" (Nothing)
          , Star
          , Parameter (TenType TensorAQ') "grad_input" (Nothing)
          , Parameter (TenType TensorAQ') "grad_weight" (Nothing)
          , Parameter (TenType TensorAQ') "grad_bias" (Nothing)
          ]
        retType nf `shouldBe` Tuple [TenType TensorA',TenType TensorA',TenType TensorA']

 where
  mhead :: [a] -> Maybe a
  mhead = \case
    [] -> Nothing
    a:_ -> Just a

  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
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

  it "parses the `_thnn_binary_cross_entropy` function without inplace and buffer" $ do
    fs <- parseWith (Proxy @ NN')

    case fmap RNN.mkBackwardAndForward $ mhead $ filter (("_thnn_binary_cross_entropy" ==) . name . NN.func') fs of
      Nothing -> fail "_thnn_binary_cross_entropy function not found!"
      Just nf -> do
        map (\v' -> Right (toTuple v')) nf `shouldBe`
          map (\s -> fmap toTuple $ parseFuncSig s)
          [ "_thnn_binary_cross_entropy_forward"
            <> "(Tensor self, Tensor target, Tensor? weight={}, int64_t reduction=Reduction::Mean)"
            <> " -> Tensor"
          , "_thnn_binary_cross_entropy_forward_out"
            <> "(Tensor output, Tensor self, Tensor target, Tensor? weight={}, int64_t reduction=Reduction::Mean)"
            <> " -> Tensor"
          , "_thnn_binary_cross_entropy_backward"
            <> "(Tensor grad_output, Tensor self, Tensor target, Tensor? weight={}, int64_t reduction=Reduction::Mean)"
            <> " -> Tensor"
          , "_thnn_binary_cross_entropy_backward_out"
            <> "(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Tensor? weight={}, int64_t reduction=Reduction::Mean)"
            <> " -> Tensor"
          ]

  it "parses the `_thnn_multilabel_margin_loss` function with buffer" $ do
    fs <- parseWith (Proxy @ NN')

    case fmap RNN.mkBackwardAndForward $ mhead $ filter (("_thnn_multilabel_margin_loss" ==) . name . NN.func') fs of
      Nothing -> fail "_thnn_multilabel_margin_loss function not found!"
      Just nf -> do
        map (\v' -> Right (toTuple v')) nf `shouldBe`
          map (\s -> fmap toTuple $ parseFuncSig s)
          [ "_thnn_multilabel_margin_loss_forward"
            <> "(Tensor self, LongTensor target, int64_t reduction=Reduction::Mean)"
            <> " -> (Tensor, Tensor)"
          , "_thnn_multilabel_margin_loss_forward_out"
            <> "(Tensor output, Tensor is_target, Tensor self, LongTensor target, int64_t reduction=Reduction::Mean)"
            <> " -> (Tensor, Tensor)"
          , "_thnn_multilabel_margin_loss_backward"
            <> "(Tensor grad_output, Tensor self, LongTensor target, int64_t reduction=Reduction::Mean, Tensor is_target)"
            <> " -> Tensor"
          , "_thnn_multilabel_margin_loss_backward_out"
            <> "(Tensor grad_input, Tensor grad_output, Tensor self, LongTensor target, int64_t reduction=Reduction::Mean, Tensor is_target)"
            <> " -> Tensor"
          ]
{-
  it "parses the `_thnn_elu` function with inplace" $ do
    fs <- parseWith (Proxy @ NN')

    case fmap RNN.mkBackwardAndForward $ mhead $ filter (("_thnn_elu" ==) . name . NN.func') fs of
      Nothing -> fail "_thnn_elu function not found!"
      Just nf -> do
        map (\v' -> Right (toTuple v')) nf `shouldBe`
          map (\s -> fmap toTuple $ parseFuncSig s)
          [ "_thnn_elu_forward"
            <> "(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)"
            <> " -> Tensor"
          , "_thnn_elu_forward_out"
            <> "(Tensor output, Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)"
            <> " -> Tensor"
          , "_thnn_elu_backward"
            <> "(Tensor grad_output, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, Tensor output)"
            <> " -> Tensor"
          , "_thnn_elu_backward_out"
            <> "(Tensor grad_input, Tensor grad_output, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, Tensor output)"
            <> " -> Tensor"
          ]
-}
  it "parses the `_thnn_rrelu_with_noise` function:generator argument is dropped.(see pytorch/aten/src/ATen/function_wrapper.py)" $ do
    fs <- parseWith (Proxy @ NN')

    case fmap RNN.mkBackwardAndForward $ mhead $ filter (("_thnn_rrelu_with_noise" ==) . name . NN.func') fs of
      Nothing -> fail "_thnn_rrelu_with_noise function not found!"
      Just nf -> do
        map (\v' -> Right (toTuple v')) nf `shouldBe`
          map (\s -> fmap toTuple $ parseFuncSig s)
          [ "_thnn_rrelu_with_noise_forward"
            <> "(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator* generator=nullptr)"
            <> " -> Tensor"
          , "_thnn_rrelu_with_noise_forward_out"
            <> "(Tensor output, Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator* generator=nullptr)"
            <> " -> Tensor"
          , "_thnn_rrelu_with_noise_backward"
            <> "(Tensor grad_output, Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false)"
            <> " -> Tensor"
          , "_thnn_rrelu_with_noise_backward_out"
            <> "(Tensor grad_input, Tensor grad_output, Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false)"
            <> " -> Tensor"
          ]

 where
  mhead :: [a] -> Maybe a
  mhead = \case
    [] -> Nothing
    a:_ -> Just a
  toTuple f = (parameters f, retType f)
  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither nnPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs


thnnPath :: FilePath
thnnPath = "../spec/THNN.h"

thcunnPath :: FilePath
thcunnPath = "../spec/THCUNN.h"

thnnSpec :: Spec
thnnSpec = do
  it "parses the same number of functions as the number of the line including `_THNN` in THNN.h" $ do
    fs <- parseWith thnnPath
    (length fs) `shouldBe` 111
  it "parses the same number of functions as the number of the line including `_THNN` in THCUNN.h" $ do
    fs <- parseWith thcunnPath
    (length fs) `shouldBe` 149
 where
  mhead :: [a] -> Maybe a
  mhead = \case
    [] -> Nothing
    a:_ -> Just a
  parseWith :: FilePath -> IO [HNN.Function]
  parseWith file = do
    readFile file >>= \f -> do
      case parse HNN.functions "" f of
        Left exception -> throw exception
        Right fs -> pure fs

derivativesPath :: FilePath
derivativesPath = "../spec/derivatives.yaml"

derivativesSpec :: Spec
derivativesSpec = do
  xs <- runIO $ vanillaParse derivativesPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ Derivative)
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither derivativesPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString "Spec doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp



declarationsPath :: FilePath
declarationsPath = "../spec/Declarations.yaml"

declarationsSpec :: Spec
declarationsSpec = do
  xs <- runIO $ vanillaParse declarationsPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ Derivative)
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither declarationsPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

