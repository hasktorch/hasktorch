module Main where

import Build_doctests (flags, pkgs, module_sources)
import System.Environment (lookupEnv)
import Test.DocTest (doctest)

main :: IO ()
main = do
    libDir <- lookupEnv "NIX_GHC_LIBDIR"

    doctest $ args ++
      maybe [] (\x -> ["-package-db " ++ x ++ "/package.conf.d"]) libDir
  where
    args = flags ++ pkgs ++ module_sources ++
      [ "-XDataKinds"
      , "-XScopedTypeVariables"
      , "-XTypeApplications"
      , "-XTypeFamilies"
      , "-fplugin GHC.TypeLits.Normalise"
      , "-fplugin GHC.TypeLits.KnownNat.Solver"
      , "-fplugin GHC.TypeLits.Extra.Solver"
      , "-fconstraint-solver-iterations=0"
      ]
