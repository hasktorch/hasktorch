module Main where

import Test.DocTest

main :: IO ()
main = doctest
  [ "-XOverloadedStrings"
  , "-XOverloadedLists"
  , "-XDataKinds"
  , "-XTypeFamilies"
  , "-XTypeApplications"
  , "-isrc"
  , "src/Torch/Static/Native"
  ]
