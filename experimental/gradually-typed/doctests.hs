module Main where

import Build_doctests (flags, module_sources, pkgs)
import Data.Foldable (traverse_)
import System.Environment (lookupEnv, setEnv)
import Test.DocTest (doctest)

main :: IO ()
main = do
  libDir <- lookupEnv "NIX_GHC_LIBDIR"
  setEnv "HASKTORCH_DEBUG" "0"

  let args =
        concat
          [ flags,
            pkgs,
            maybe [] (\x -> ["-package-db " <> x <> "/package.conf.d"]) libDir,
            [ "-XDataKinds",
              "-XPatternSynonyms",
              "-XScopedTypeVariables",
              "-XTypeApplications",
              "-XTypeFamilies",
              "-XFlexibleContexts",
              "-XQuasiQuotes",
              "-XTemplateHaskell",
              "-XHaskell2010",
              "-fplugin GHC.TypeLits.Normalise",
              "-fplugin GHC.TypeLits.KnownNat.Solver",
              "-fplugin GHC.TypeLits.Extra.Solver",
              "-fconstraint-solver-iterations=0"
            ],
            module_sources
          ]

  -- traverse_ putStrLn args
  doctest args
