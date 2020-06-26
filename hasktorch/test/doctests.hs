module Main where

import Test.DocTest

import Build_doctests (flags, pkgs, module_sources)

main :: IO ()
main = do
  print pkgs
  print flags
  print module_sources
  doctest $
    [ "-XDataKinds"
    , "-XScopedTypeVariables"
    , "-XTypeApplications"
    , "-XTypeFamilies"
    , "-fplugin GHC.TypeLits.Normalise"
    , "-fplugin GHC.TypeLits.KnownNat.Solver"
    , "-fplugin GHC.TypeLits.Extra.Solver"
    , "-fconstraint-solver-iterations=0"
    -- , "-isrc"
    -- , "src/Torch/Typed/Tensor"
    -- , "src/Torch/Typed/Factories"
    -- , "src/Torch/Typed/Functional"
    -- , "src/Torch/Typed/NN/Recurrent/LSTM"
    -- , "src/Torch/Typed/NN/Recurrent/GRU"
    ]
    <> flags
    <> pkgs
    <> module_sources
