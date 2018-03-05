{-# LANGUAGE OverloadedStrings #-}
module FileMappings where

import Data.Monoid ((<>))
import Data.Text (Text)
import qualified Data.Text as T

import CodeGen.Types -- (HModule, TemplateType, genericTypes)
import RenderShared (makeModule) -- , renderCHeaderFile, parseFile, IsTemplate(..))

type HeaderFile = FilePath

thFiles :: CodeGenType -> [(String, TemplateType -> [THFunction] -> HModule)]
thFiles = \case
  GenericFiles -> map ($ GenericFiles)
    [ mkTHGeneric' "Blas"
    , mkTHGeneric' "Lapack"
    , mkTHGeneric' "Storage"
    , mkTHGeneric  (ModuleSuffix "Storage") (FileSuffix "StorageCopy")
    , mkTHGeneric' "Tensor"
    , mkTHGeneric  (ModuleSuffix "Tensor") (FileSuffix "TensorConv")
    , mkTHGeneric  (ModuleSuffix "Tensor") (FileSuffix "TensorCopy")
    , mkTHGeneric  (ModuleSuffix "Tensor") (FileSuffix "TensorLapack")
    , mkTHGeneric  (ModuleSuffix "Tensor") (FileSuffix "TensorMath")
    , mkTHGeneric  (ModuleSuffix "Tensor") (FileSuffix "TensorRandom")
    , mkTHGeneric' "Vector"
    ]
  ConcreteFiles -> map ($ ConcreteFiles)
    [ mkTHFile' "File"
    , mkTHFile' "DiskFile"
    , mkTHFile' "Atomic"
    , mkTHFile' "Half"
    , mkTHFile' "LogAdd"
    , mkTHFile' "Random"
    , mkTHFile' "Size"
    , mkTHFile' "Storage"
    , mkTHFile' "MemoryFile"
    ]
  ManagedFiles -> [] -- FIXME: check with austin that managed codegen is still a thing we want to do

 where
  out :: CodeGenType -> TextPath
  out = TextPath . T.pack . outDir TH

  src :: CodeGenType -> FilePath
  src = srcDir TH

  mkTuple
    :: LibType
    -> IsTemplate
    -> ModuleSuffix
    -> FileSuffix
    -> CodeGenType
    -> (FilePath, TemplateType -> [THFunction] -> HModule)
  mkTuple lt b modsuff filesuff cgt
    = (src cgt <> hf, makeModule TH (out cgt) b hf modsuff filesuff)
   where
    hf :: FilePath
    hf = show lt <> T.unpack (textFileSuffix filesuff) <> ".h"

  mkTHFile, mkTHGeneric, mkTHManaged
    :: ModuleSuffix
    -> FileSuffix
    -> CodeGenType
    -> (FilePath, TemplateType -> [THFunction] -> HModule)
  mkTHFile = mkTuple TH asfile
  mkTHGeneric = mkTuple TH astemplate
  mkTHManaged = mkTuple TH astemplate

  mkTHFile', mkTHGeneric', mkTHManaged'
    :: Text
    -> CodeGenType
    -> (FilePath, TemplateType -> [THFunction] -> HModule)
  mkTHFile' suff = mkTHFile (ModuleSuffix suff) (FileSuffix suff)
  mkTHGeneric' suff = mkTHGeneric (ModuleSuffix suff) (FileSuffix suff)
  mkTHManaged' suff = mkTHManaged (ModuleSuffix suff) (FileSuffix suff)

astemplate = IsTemplate True
asfile = IsTemplate False
