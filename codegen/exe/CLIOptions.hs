{-# LANGUAGE ScopedTypeVariables #-}
module CLIOptions where

import Options.Applicative as OptParse
import Data.List (intercalate)
import Data.Monoid ((<>))
import Data.Proxy (Proxy(..))

import CodeGen.Types

-- ========================================================================= --

-- | CLI Options
--
-- FIXME: Allow taking @HashSet <types>@ so that we can run multiple targets at
-- the same time.
data Options = Options
  { codegenType :: CodeGenType
  , libraries   :: LibType
  , verbose     :: Bool
  }


-- | optparse-applicative Parser to annotate and parse our CLI
cliOptions :: OptParse.Parser Options
cliOptions = Options
  <$> option auto
      ( long "type"
    <> short 't'
    <> help "which type of codegen to run"
    <> metavar (enumVar (Proxy :: Proxy CodeGenType) generatable))
  <*> option auto
      ( long "lib"
    <> short 'l'
    <> help "which library to run against"
    <> metavar (enumVar (Proxy :: Proxy LibType) supported))
  <*> flag' False
      ( long "verbose"
    <> short 'v'
    <> help "whether or not to print debugging informations")
 where
  enumVar
    :: forall a . (Bounded a, Enum a, Show a)
    => Proxy a -> (a -> Bool) -> String
  enumVar _ f
    = "[" ++ intercalate "|" (show <$> filter f [minBound..maxBound::a]) ++ "]"



