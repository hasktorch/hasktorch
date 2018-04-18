{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
module RawLapackSVDSpec where

import Foreign.C.Types

import Torch.Dimensions
import Torch.Dynamic

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: pcaRaw" pcaRaw

pcaRaw :: IO ()
pcaRaw = do

  a :: FloatDynamic <- constant (dim :: Dim '[2, 2]) 2
  printTensor a

  b                 <- constant (dim :: Dim '[2]) 1
  printTensor b

  _set2d a 0 0 1.0
  _set2d a 0 1 2.0
  _set2d a 1 0 3.0
  _set2d a 1 0 4.0
  printTensor a
  printTensor b

  resA <- constant (dim :: Dim '[2, 2]) 0
  resB <- constant (dim :: Dim '[2, 2]) 0
  _gesv resB resA b a
  printTensor resA
  printTensor resB

