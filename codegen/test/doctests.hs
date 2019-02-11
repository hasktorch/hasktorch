module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [
      "-XOverloadedStrings",
      "-package=megaparsec",
      "-package-db=../.stack-work/install/x86_64-linux/lts-13.1/8.6.3/pkgdb/",
      "src/ParseFunctionSig.hs"
    ]
