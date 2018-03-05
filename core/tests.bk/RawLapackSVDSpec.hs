{-# LANGUAGE DataKinds #-}
module RawLapackSVDSpec where

import Foreign.C.Types

import Torch.FFI.TH.Double.Tensor
import Torch.FFI.TH.Double.TensorLapack
import Torch.FFI.TH.Double.TensorMath
import Torch.FFI.TH.Double.TensorRandom

import Torch.Raw.Tensor.Generic (dispRaw, constant)
import Torch.Dimensions

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: pcaRaw" pcaRaw

pcaRaw :: IO ()
pcaRaw = do
  a <- constant (dim :: Dim '[2, 2]) 2.0
  b <- constant (dim :: Dim '[2]) 1.0
  c_Torch.FFI.TH.Double.Tensor_set2d a 0 0 1.0
  c_Torch.FFI.TH.Double.Tensor_set2d a 0 1 2.0
  c_Torch.FFI.TH.Double.Tensor_set2d a 1 0 3.0
  c_Torch.FFI.TH.Double.Tensor_set2d a 1 0 4.0
  dispRaw a
  dispRaw b
  resA <- constant (dim :: Dim '[2, 2]) 0.0
  resB <- constant (dim :: Dim '[2, 2]) 0.0
  c_Torch.FFI.TH.Double.Tensor_gesv resB resA b a
  dispRaw resA
  dispRaw resB
  c_Torch.FFI.TH.Double.Tensor_free a
  c_Torch.FFI.TH.Double.Tensor_free b
  c_Torch.FFI.TH.Double.Tensor_free resA
  c_Torch.FFI.TH.Double.Tensor_free resB
  pure ()

