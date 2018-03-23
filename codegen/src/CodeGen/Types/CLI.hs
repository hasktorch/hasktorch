{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveDataTypeable #-}
module CodeGen.Types.CLI
  ( LibType(..)
  , prefix
  , describe'
  , supported
  , supportedLibraries
  , outDir
  , outModule
  , srcDir
  , CodeGenType(..)
  , generatable
  , TemplateType(..)
  , generatedTypes
  ) where

import Data.Data
import Data.Typeable
import CodeGen.Prelude
import qualified Data.HashSet as HS


-- | All possible libraries that we intend to support (these are all src
-- libraries in ATen). Note that this ordering is used in codegen and must not be changed.
data LibType
  = ATen
  | THCUNN
  | THCS
  | THC
  | THNN
  | THS
  | TH
  deriving (Eq, Ord, Show, Enum, Bounded, Read, Generic, Hashable, Data, Typeable)


prefix :: LibType -> Bool -> Text
prefix lt long =
  case lt of
    THC -> if long then "THCuda" else "THC"
    _   -> tshow lt


-- | Short descriptions of each library we intend to support.
describe' :: LibType -> String
describe' = \case
  ATen -> "A simple TENsor library thats exposes the Tensor operations in Torch"
       ++ "and PyTorch directly in C++11."
  TH -> "Torch7"
  THC -> "Cuda-based Torch7"
  THCS -> "Cuda-based Sparse Tensor support with TH"
  THCUNN -> "Cuda-based THNN"
  THNN -> "THNN"
  THS -> "TH Sparse tensor support (ATen library)"


-- | Whether or not we currently support code generation for the library
supported :: LibType -> Bool
supported lt = lt `HS.member` HS.fromList [TH, THC, THNN, THCUNN]

supportedLibraries :: [LibType]
supportedLibraries = filter supported [minBound..maxBound]

-- | Where generated code will be placed.
outDir :: LibType -> FilePath
outDir lt = intercalate ""
  [ "output/raw/"
  , toLowers lt ++ "/"
  , "src/"
  -- , if cgt == GenericFiles then "generic/" else ""
  , "Torch/FFI/" ++ show lt
  ]
 where
  toLowers :: Show a => a -> String
  toLowers = map toLower . show


-- | The prefix of the output module name
outModule :: LibType -> Text
outModule lt = "Torch.FFI." <> tshow lt


-- | Where the source files are located, relative to the root of the hasktorch
-- project.
srcDir :: LibType -> CodeGenType -> FilePath
srcDir lt cgt = intercalate ""
  [ "./vendor/pytorch/aten/src/"
  , show lt ++ "/"
  , if cgt == GenericFiles then "generic/" else ""
  ]


-- | Type of code to generate
data CodeGenType
  = GenericFiles   -- ^ generic/ files which are used in C for type-generic code
  | ConcreteFiles  -- ^ concrete supporting files. These include utility
                   --   functions and random generators.
  deriving (Eq, Ord, Enum, Bounded)

instance Read CodeGenType where
  readsPrec _ s = case s of
    "generic"  -> [(GenericFiles, "")]
    "concrete" -> [(ConcreteFiles, "")]
    _          -> []

instance Show CodeGenType where
  show = \case
    GenericFiles  -> "generic"
    ConcreteFiles -> "concrete"


-- | Whether or not we currently support generating this type of code (ie: I
-- (\@stites) am not sure about the managed files).
generatable :: CodeGenType -> Bool
generatable = const True

-- ----------------------------------------
-- Types for representing templating
-- ----------------------------------------

data TemplateType
  = GenByte
  | GenChar
  | GenDouble
  | GenFloat
  | GenHalf
  | GenInt
  | GenLong
  | GenShort
  | GenNothing
  deriving (Eq, Ord, Bounded, Show, Generic, Hashable)


-- List used to iterate through all template types
generatedTypes :: LibType -> CodeGenType -> [TemplateType]
generatedTypes THNN   = \case { ConcreteFiles -> [GenNothing]; GenericFiles -> [GenDouble, GenFloat] }
generatedTypes THCUNN = \case { ConcreteFiles -> [GenNothing]; GenericFiles -> [GenDouble, GenFloat] }
generatedTypes _ = \case
  ConcreteFiles -> [GenNothing]
  GenericFiles ->
    [ GenByte
    , GenChar
    , GenDouble
    , GenFloat
    , GenHalf
    , GenInt
    , GenLong
    , GenShort
    ]

