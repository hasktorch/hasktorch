module Main where

import           Test.DocTest

main :: IO ()
main = doctest
  [ "-XOverloadedStrings"
  , "-XDataKinds"
  , "-XTypeFamilies"
  , "-XTypeApplications"
      -- , "-isrc"
  , "src/Torch/Cast.hs"
  , "src/Torch/Layout.hs"
  , "src/Torch/Backend.hs"
  , "src/Torch/Scalar.hs"
  , "src/Torch/DType.hs"
  , "src/Torch/TensorOptions.hs"
  , "src/Torch/Tensor.hs"
  , "src/Torch/TensorFactories.hs"
  , "src/Torch/Autograd.hs"
  , "src/Torch/Functions/Native.hs"
  , "src/Torch/Functions.hs"
  , "src/Torch/NN.hs"
  , "src/Torch/Static/Factories.hs"
  , "src/Torch/Static.hs"
  , "src/Torch/Static/Native.hs"
  ]
