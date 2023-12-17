{-# LANGUAGE NoMonomorphismRestriction #-}

module FunctionalSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
--import Torch.Tensor
--import Torch.DType
--import Torch.TensorFactories
--import Torch.Functional
--import Torch.TensorOptions
import Torch
import Prelude hiding (abs, all, div, exp, floor, log, max, min)

spec :: Spec
spec = do
  it "scales and adds" $ do
    let x = 2 * ones' [10] + 3 * ones' [10]
    (toDouble $ select 0 4 x) `shouldBe` 5.0
  it "sumAll" $ do
    let x = sumAll (2 * ones' [5])
    toDouble x `shouldBe` 10.0
  it "abs" $ do
    let x = abs $ (-2) * ones' [5]
    (toDouble $ select 0 0 x) `shouldBe` 2.0
  it "add" $ do
    let x = (-2) * ones' [5]
    let y = abs x
    let z = add x y
    (toDouble $ select 0 0 z) `shouldBe` 0.0
  it "sub" $ do
    let x = (-2) * ones' [5]
    let y = abs x
    let z = sub x y
    (toDouble $ select 0 0 z) `shouldBe` -4.0
  it "mul" $ do
    let x = (-5) * ones' [5]
    let y = 2 * ones' [5]
    let z = mul x y
    (toDouble $ select 0 0 z) `shouldBe` -10.0
  it "div" $ do
    let x = (-5) * ones' [5]
    let y = 2 * ones' [5]
    let z = div x y
    (toDouble $ select 0 0 z) `shouldBe` -2.5
  it "ceil" $ do
    x <- randIO' [5]
    let y = ceil x
    (toDouble $ select 0 0 y) `shouldBe` 1.0
  it "floor" $ do
    x <- randIO' [5]
    let y = floor x
    (toDouble $ select 0 0 y) `shouldBe` 0.0
  it "takes the minimum of a linspace" $ do
    let x = linspace (5.0 :: Double) (25.0 :: Double) 50 defaultOpts
    let m = min x
    toDouble m `shouldBe` 5.0
  it "takes the maximum of a linspace" $ do
    let x = linspace (5.0 :: Double) (25.0 :: Double) 50 defaultOpts
    let m = max x
    toDouble m `shouldBe` 25.0
  it "takes the median of a linspace" $ do
    let x = linspace (5.0 :: Double) (10.0 :: Double) 5 defaultOpts
    let m = median x
    toDouble m `shouldBe` 7.5
  it "performs matrix vector multiplication" $ do
    let m = 3 * ones' [5, 5]
    let v = 2 * ones' [5, 1]
    let x = matmul m v
    (toDouble $ select 0 0 x) `shouldBe` 30.0
  it "erf" $ do
    let x = erf $ zeros' [4]
    (toDouble $ select 0 0 x) `shouldBe` 0.0
  it "exp" $ do
    let x = exp $ zeros' [4]
    (toDouble $ select 0 0 x) `shouldBe` 1.0
  it "log1p" $ do
    let x = log1p $ zeros' [4]
    (toDouble $ select 0 0 x) `shouldBe` 0.0
  it "log2" $ do
    let x = log2 $ 4 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 2.0
  it "log10" $ do
    let x = log10 $ 1000 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 3.0
  it "relu (pos)" $ do
    let x = relu $ 5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 5.0
  it "relu (neg)" $ do
    let x = relu $ -5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 0.0
  {-
   gels is deprecated. use lstsq.
   -- deps/pytorch/torch/functional.py --
    .. warning::
        :func:`torch.gels` is deprecated in favour of :func:`torch.lstsq` and will be removed in the
        next release. Please use :func:`torch.lstsq` instead.
  -}
  it "lstsq" $ do
    let x = lstsq (ones' [5, 2]) (ones' [5, 3])
    shape x `shouldBe` [3, 2]
  it "diag" $ do
    let x = ones' [3]
    let y = diag (Diag 2) x
    shape y `shouldBe` [5, 5]
  it "diagEmbed" $ do
    let t = ones' [2, 3]
    shape (diagEmbed (Diag 0) (Dim (-2)) (Dim (-1)) t) `shouldBe` [2, 3, 3]
    shape (diagEmbed (Diag 1) (Dim 0) (Dim 2) t) `shouldBe` [4, 2, 4]
  it "diagflat" $ do
    let t1 = ones' [3]
    shape (diagflat (Diag 0) t1) `shouldBe` [3, 3]
    shape (diagflat (Diag 1) t1) `shouldBe` [4, 4]
    let t2 = ones' [2, 2]
    shape (diagflat (Diag 0) t2) `shouldBe` [4, 4]
  it "diagonal" $ do
    let t1 = ones' [3, 3]
    shape (diagonal (Diag 0) (Dim 0) (Dim 1) t1) `shouldBe` [3]
    shape (diagonal (Diag 1) (Dim 0) (Dim 1) t1) `shouldBe` [2]
    let t2 = ones' [2, 5, 4, 2]
    shape (diagonal (Diag (-1)) (Dim 1) (Dim 2) t2) `shouldBe` [2, 2, 4]
  it "expand" $ do
    let t = asTensor [[1], [2], [3 :: Int]]
    shape (expand t False [3, 4]) `shouldBe` [3, 4]
  it "flattenAll" $ do
    let t = asTensor [[1, 2], [3, 4 :: Int]]
    shape (flattenAll t) `shouldBe` [4]

  -- decomposition / solvers
  it "solve" $ do
    a <- randIO' [10, 10]
    b <- randIO' [10, 3]
    let x = solve b a
    shape x `shouldBe` [10, 3]

  it "cholesky decomposes" $ do
    let x = asTensor ([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]] :: [[Double]])
        c = cholesky Upper x
        c' = asTensor ([[2.0, 6.0, -8.0], [0.0, 1.0, 5.0], [0.0, 0.0, 3.0]] :: [[Double]])
    all (c ==. c') `shouldBe` True
  it "inverse of an identity matrix is an identity matrix" $ do
    let soln = eq (inverse $ eye' 3 3) (eye' 3 3)
    all soln `shouldBe` True
  it "conv1d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel = 1
        input = 5
        x =
          conv1d'
            (ones' [out_channel, in_channel, kernel])
            (ones' [out_channel])
            1
            0
            (ones' [batch, in_channel, input])
    shape x `shouldBe` [batch, out_channel, input]
  it "conv2d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel0 = 1
        kernel1 = 1
        input0 = 5
        input1 = 6
        x =
          conv2d'
            (ones' [out_channel, in_channel, kernel0, kernel1])
            (ones' [out_channel])
            (1, 1)
            (0, 0)
            (ones' [batch, in_channel, input0, input1])
    shape x `shouldBe` [batch, out_channel, input0, input1]
  it "conv3d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel0 = 1
        kernel1 = 1
        kernel2 = 1
        input0 = 5
        input1 = 6
        input2 = 7
        x =
          conv3d'
            (ones' [out_channel, in_channel, kernel0, kernel1, kernel2])
            (ones' [out_channel])
            (1, 1, 1)
            (0, 0, 0)
            (ones' [batch, in_channel, input0, input1, input2])
    shape x `shouldBe` [batch, out_channel, input0, input1, input2]
  it "convTranspose1d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel = 1
        input = 5
        x =
          convTranspose1d'
            (ones' [in_channel, out_channel, kernel])
            (ones' [out_channel])
            1
            0
            (ones' [batch, in_channel, input])
    shape x `shouldBe` [batch, out_channel, input]
  it "convTranspose2d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel0 = 1
        kernel1 = 1
        input0 = 5
        input1 = 6
        x =
          convTranspose2d'
            (ones' [in_channel, out_channel, kernel0, kernel1])
            (ones' [out_channel])
            (1, 1)
            (0, 0)
            (ones' [batch, in_channel, input0, input1])
    shape x `shouldBe` [batch, out_channel, input0, input1]
  it "convTranspose3d" $ do
    let batch = 10
        in_channel = 3
        out_channel = 10
        kernel0 = 1
        kernel1 = 1
        kernel2 = 1
        input0 = 5
        input1 = 6
        input2 = 7
        x =
          convTranspose3d'
            (ones' [in_channel, out_channel, kernel0, kernel1, kernel2])
            (ones' [out_channel])
            (1, 1, 1)
            (0, 0, 0)
            (ones' [batch, in_channel, input0, input1, input2])
    shape x `shouldBe` [batch, out_channel, input0, input1, input2]
  it "elu (pos)" $ do
    let x = elu (0.5 :: Float) $ 5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 5.0
  it "elu (neg)" $ do
    let x = elu (0.5 :: Float) $ -5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` (-0.49663102626800537)
  it "elu' (pos)" $ do
    let x = elu' $ 5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` 5.0
  it "elu' (neg)" $ do
    let x = elu' $ -5 * ones' [4]
    (toDouble $ select 0 0 x) `shouldBe` (-0.9932620525360107)
  it "embedding" $ do
    let dic = asTensor ([[1, 2, 3], [4, 5, 6]] :: [[Float]])
        indices = asTensor ([0, 1, 1] :: [Int])
        x = embedding' dic indices
        value = asTensor ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]] :: [[Float]])
    Torch.all (x `eq` value) `shouldBe` True
  it "smoothL1Loss" $ do
    let input = ones' [3]
        target = 3 * input
        output = smoothL1Loss ReduceNone input target
    (toDouble $ select 0 0 output) `shouldBe` (1.5)
  it "softMarginLoss" $ do
    let input = ones' [3]
        target = 3 * input
        output = softMarginLoss ReduceSum input target
    (toInt $ output * 1000) `shouldBe` (145)
  it "softShrink" $ do
    let input = 3 * ones' [3]
        output = softShrink 1 input
    (toDouble $ select 0 0 output) `shouldBe` (2.0)
  it "stack" $ do
    let x = ones' [4, 3]
        y = ones' [4, 3]
        output = stack (Dim 1) [x, y]
    (shape output) `shouldBe` ([4, 2, 3])
  it "sumDim" $ do
    let x = asTensor ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] :: [[Float]])
        output = sumDim (Dim 0) KeepDim Float x
    (toDouble $ select 1 1 output) `shouldBe` (26.0)
  it "topK" $ do
    let x = asTensor ([1, 2, 3] :: [Float])
        output = fst $ topK 2 (Dim 0) True True x
    (toDouble $ select 0 0 output) `shouldBe` (3.0)
  it "triu" $ do
    let x = asTensor ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] :: [[Float]])
    (toDouble $ sumAll $ triu (Diag 0) x) `shouldBe` (26.0)
  it "tril" $ do
    let x = asTensor ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] :: [[Float]])
    (toDouble $ sumAll $ tril (Diag 0) x) `shouldBe` (67.0)
  it "unsqueeze" $ do
    let x = asTensor ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] :: [[Float]])
        output = unsqueeze (Dim 0) x
    (shape output) `shouldBe` ([1, 4, 3])
  it "ctcLoss" $ do
    ctcLoss'
      ReduceMean
      [1]
      [1]
      (asTensor ([[[0.1, 0.2, 0.7]]] :: [[[Float]]]))
      (asTensor ([2] :: [Int]))
      `shouldBe` asTensor (-0.7 :: Float)
