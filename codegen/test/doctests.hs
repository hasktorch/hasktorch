module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [
      "-XOverloadedStrings",
      "ParseFunctionSig.hs"
    ]
