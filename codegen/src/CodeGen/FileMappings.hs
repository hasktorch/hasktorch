{-# LANGUAGE OverloadedStrings #-}
module CodeGen.FileMappings
  ( files
  , HeaderFile
  ) where

import Data.Monoid ((<>))
import Data.Text (Text)
import qualified Data.Text as T

import CodeGen.Types
import CodeGen.Render (makeModule)

type HeaderFile = FilePath

files :: LibType -> CodeGenType -> [(String, TemplateType -> [Function] -> HModule)]
files TH = \case
  GenericFiles -> map (\fn -> fn TH GenericFiles)
    [ mkModule' "Blas"
    , mkModule' "Lapack"
    , mkModule' "Storage"
    , mkModule  (ModuleSuffix "Storage") "StorageCopy"
    , mkModule' "Tensor"
    , mkModule  (ModuleSuffix "Tensor") "TensorConv"
    , mkModule  (ModuleSuffix "Tensor") "TensorCopy"
    , mkModule  (ModuleSuffix "Tensor") "TensorLapack"
    , mkModule  (ModuleSuffix "Tensor") "TensorMath"
    , mkModule  (ModuleSuffix "Tensor") "TensorRandom"
    , mkModule' "Vector"
    ]
  ConcreteFiles -> map (\fn -> fn TH ConcreteFiles)
    [ mkModule' "File"
    , mkModule' "DiskFile"
    , mkModule' "Atomic"
    , mkModule' "Half"
    , mkModule' "LogAdd"
    , mkModule' "Random"
    , mkModule' "Size"
    , mkModule' "Storage"
    , mkModule' "MemoryFile"
    ]

files THC = \case
  GenericFiles -> map (\fn -> fn THC GenericFiles)
    [ mkModule' "Storage"
    , mkModule  (ModuleSuffix "Storage") "StorageCopy"
    , mkModule' "Tensor"
    , mkModule  (ModuleSuffix "Tensor") "TensorCopy"
    , mkModule  (ModuleSuffix "Tensor") "TensorIndex"
    , mkModule  (ModuleSuffix "Tensor") "TensorMasked"
    , mkModule  (ModuleSuffix "Tensor") "TensorMath"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathBlas"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathCompare"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathCompareT"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathMagma"     --  NOTE: CUDA implementation of LAPACK functions
    , mkModule  (ModuleSuffix "Tensor") "TensorMathPairwise"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathReduce"
    , mkModule  (ModuleSuffix "Tensor") "TensorMathScan"
    , mkModule  (ModuleSuffix "Tensor") "TensorMode"
    , mkModule  (ModuleSuffix "Tensor") "TensorRandom"
    , mkModule  (ModuleSuffix "Tensor") "TensorScatterGather"
    , mkModule  (ModuleSuffix "Tensor") "TensorSort"
    , mkModule  (ModuleSuffix "Tensor") "TensorTopK"
    ]
  -- does not account for any .cuh files
  ConcreteFiles -> map (\fn -> fn THC ConcreteFiles)
    [ mkModule' "Allocator"
    , mkModule' "Blas"
    , mkModule' "CachingAllocator"
    , mkModule' "CachingHostAllocator"
    -- , mkModule' "THCDeviceTensor" -- this is a .cuh
    , mkModule' "Half"
    -- , mkModule' "THCReduce" -- this is a .cuh
    , mkModule' "Sleep"
    , mkModule' "Storage"
    , mkModule' "StorageCopy"
    , mkModule' "Stream"
    , mkModule' "Tensor"
    , mkModule' "TensorConv"
    , mkModule' "TensorCopy"
    , mkModule' "TensorMath"
    , mkModule' "TensorRandom"
    , mkModule' "ThreadLocal"
    ]
files _ = \case {ConcreteFiles -> []; GenericFiles -> []}


mkModule
  :: ModuleSuffix
  -> FileSuffix
  -> LibType
  -> CodeGenType
  -> (FilePath, TemplateType -> [Function] -> HModule)
mkModule modsuff filesuff lt cgt
  = (srcDir lt cgt <> hf, makeModule TH (TextPath . T.pack $ outDir lt) cgt hf modsuff filesuff)
 where
  hf :: FilePath
  hf = show lt <> T.unpack (textFileSuffix filesuff) <> ".h"


mkModule'
  :: Text
  -> LibType
  -> CodeGenType
  -> (FilePath, TemplateType -> [Function] -> HModule)
mkModule' suff = mkModule (ModuleSuffix suff) (FileSuffix suff)

