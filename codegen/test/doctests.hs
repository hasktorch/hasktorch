module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [ "-XOverloadedStrings",
      "src/Codegen/ParseFunctionSig.hs"
    ]
