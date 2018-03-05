{-# LANGUAGE OverloadedStrings #-}
module FileMappings where

import Data.Monoid ((<>))
import Data.Text (Text)
import qualified Data.Text as T

import CodeGenParse (THFunction, Parser, thParseGeneric)
import CodeGen.Types -- (HModule, TemplateType, genericTypes)
import RenderShared (makeModule, renderCHeaderFile, parseFile, IsTemplate(..))

import CLIOptions

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

  ConcreteFiles ->
    [ (src ConcreteFiles <> "THFile.h"        , (makeModule TH (out ConcreteFiles) asfile "THFile.h" "File" "File"))
    , (src ConcreteFiles <> "THDiskFile.h"    , (makeModule TH (out ConcreteFiles) asfile "THDiskFile.h" "DiskFile" "DiskFile"))
    , (src ConcreteFiles <> "THAtomic.h"      , (makeModule TH (out ConcreteFiles) asfile "THDiskFile.h" "Atomic" "Atomic"))
    , (src ConcreteFiles <> "THHalf.h"        , (makeModule TH (out ConcreteFiles) asfile "THHalf.h" "Half" "Half"))
    , (src ConcreteFiles <> "THLogAdd.h"      , (makeModule TH (out ConcreteFiles) asfile "THLogAdd.h" "LogAdd" "LogAdd"))
    , (src ConcreteFiles <> "THRandom.h"      , (makeModule TH (out ConcreteFiles) asfile "THRandom.h" "Random" "Random"))
    , (src ConcreteFiles <> "THSize.h"        , (makeModule TH (out ConcreteFiles) asfile "THSize.h" "Size" "Size"))
    , (src ConcreteFiles <> "THStorage.h"     , (makeModule TH (out ConcreteFiles) asfile "THStorage.h" "Storage" "Storage"))
    , (src ConcreteFiles <> "THMemoryFile.h"  , (makeModule TH (out ConcreteFiles) asfile "THMemoryFile.h" "MemoryFile" "MemoryFile"))
    ]
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

  mkTHFile, mkTHGeneric
    :: ModuleSuffix
    -> FileSuffix
    -> CodeGenType
    -> (FilePath, TemplateType -> [THFunction] -> HModule)
  mkTHFile = mkTuple TH asfile
  mkTHGeneric = mkTuple TH astemplate

  mkTHFile', mkTHGeneric'
    :: Text
    -> CodeGenType
    -> (FilePath, TemplateType -> [THFunction] -> HModule)
  mkTHFile' suff = mkTHFile (ModuleSuffix suff) (FileSuffix suff)
  mkTHGeneric' suff = mkTHGeneric (ModuleSuffix suff) (FileSuffix suff)

astemplate = IsTemplate True
asfile = IsTemplate False
