module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [
      "-XOverloadedStrings",
      "-XDataKinds",
      "-XTypeFamilies",
      "-XTypeApplications",
      "src/Torch/Static/Native.hs"
    ]
