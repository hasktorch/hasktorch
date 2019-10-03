module Main where

import Test.DocTest

main :: IO ()
main = doctest
  [ "-XOverloadedStrings"
  , "-XOverloadedLists"
  , "-XDataKinds"
  , "-XTypeFamilies"
  , "-XTypeApplications"
  , "-fplugin GHC.TypeLits.Normalise"
  , "-fplugin GHC.TypeLits.KnownNat.Solver"
  , "-fplugin GHC.TypeLits.Extra.Solver"
  , "-fconstraint-solver-iterations=0"
  , "-isrc"
  , "src/Torch/Static/Native"
  ]
