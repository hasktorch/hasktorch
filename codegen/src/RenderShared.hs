{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module RenderShared (
  makeModule
  ) where

import CodeGenTypes

makeModule modHeader modSuffix modFileSuffix typeTemplate bindings =
   HModule {
        modHeader = modHeader,
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = modSuffix,
        modFileSuffix = modFileSuffix,
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }
