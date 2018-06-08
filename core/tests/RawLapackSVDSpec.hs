{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
module RawLapackSVDSpec where

import Foreign.C.Types

import Numeric.Dimensions
import Torch.Float.Dynamic

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: pcaRaw" pcaRaw

pcaRaw :: IO ()
pcaRaw = do

  let a :: FloatDynamic = constant (dims :: Dims '[2, 2]) 2
  print a

  let b                 = constant (dims :: Dims '[2]) 1
  print b

  _set2d a 0 0 1.0
  _set2d a 0 1 2.0
  _set2d a 1 0 3.0
  _set2d a 1 0 4.0
  print a
  print b

  let resA = constant (dims :: Dims '[2, 2]) 0
  let resB = constant (dims :: Dims '[2, 2]) 0
  gesv_ (resB,resA) b a
  print resA
  print resB

