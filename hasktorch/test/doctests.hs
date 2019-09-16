module Main where

import Test.DocTest

main :: IO ()
main = doctest
  [ "-XOverloadedStrings"
  , "-XDataKinds"
  , "-XTypeFamilies"
  , "-XTypeApplications"
  , "-isrc"
  , "src/Torch/Static/Native"
  ]
