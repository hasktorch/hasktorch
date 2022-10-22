module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [ "-XOverloadedStrings",
      "-XQuasiQuotes",
      "-XTemplateHaskell",
      "-XScopedTypeVariables",
      "-XHaskell2010",
      "src/ParseFunctionSig.hs"
    ]
