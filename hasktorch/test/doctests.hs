module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [
      "-XOverloadedStrings",
      "-XDataKinds",
      "-XTypeFamilies",
      "src/Torch/Static/Native.hs"
    ]
