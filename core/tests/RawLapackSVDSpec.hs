module RawLapackSVDSpec where

import Foreign.C.Types

import THDoubleTensor
import THDoubleTensorLapack
import THDoubleTensorMath
import THDoubleTensorRandom

import Torch.Core.Tensor.Types (TensorDim(D1, D2))
import Torch.Core.Tensor.Raw (dispRaw, tensorRaw)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: pcaRaw" pcaRaw

pcaRaw :: IO ()
pcaRaw = do
  a <- tensorRaw (D2 (2, 2)) 2.0
  b <- tensorRaw (D1 2) 1.0
  c_THDoubleTensor_set2d a 0 0 1.0
  c_THDoubleTensor_set2d a 0 1 2.0
  c_THDoubleTensor_set2d a 1 0 3.0
  c_THDoubleTensor_set2d a 1 0 4.0
  dispRaw a
  dispRaw b
  resA <- tensorRaw (D2 (2, 2)) 0.0
  resB <- tensorRaw (D2 (2, 2)) 0.0
  c_THDoubleTensor_gesv resB resA b a
  dispRaw resA
  dispRaw resB
  c_THDoubleTensor_free a
  c_THDoubleTensor_free b
  c_THDoubleTensor_free resA
  c_THDoubleTensor_free resB
  pure ()

