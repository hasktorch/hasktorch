{-# LANGUAGE CPP #-}

module GHC.NotExport.Plugin (plugin) where

#if MIN_VERSION_ghc(9,0,0)
import GHC.Driver.Plugins
import GHC.Types.Avail
import GHC.Types.Name
import GHC.Tc.Types
import GHC.Utils.Outputable
#else
import GhcPlugins
import Plugins
import TcRnTypes
import Avail
#endif

import Data.List (isPrefixOf)

plugin :: Plugin
plugin =
  defaultPlugin
    { typeCheckResultAction = notExportPlugins,
      pluginRecompile = purePlugin
    }

notExportPlugins cmdOptions modSummary env = do
  let updated_tcg_exports = filter (\v -> not (isPrefixOf "inline_c_ffi" ((showSDocUnsafe . ppr . nameOccName . availName) v))) $ tcg_exports env
  return env {tcg_exports = updated_tcg_exports}
