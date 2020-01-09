module Main where

import Test.DocTest

main :: IO ()
main = doctest
  [ "-XOverloadedStrings"
  , "-XOverloadedLists"
  , "-XDataKinds"
  , "-XScopedTypeVariables"
  , "-XTypeFamilies"
  , "-XTypeApplications"
  , "-fplugin GHC.TypeLits.Normalise"
  , "-fplugin GHC.TypeLits.KnownNat.Solver"
  , "-fplugin GHC.TypeLits.Extra.Solver"
  , "-fconstraint-solver-iterations=0"
  , "-isrc"
  , "src/Torch/Typed/Tensor"
  , "src/Torch/Typed/Factories"
  , "src/Torch/Typed/Functional"
  , "src/Torch/Typed/NN/Recurrent/LSTM"
  , "src/Torch/Typed/NN/Recurrent/GRU"
  ]
