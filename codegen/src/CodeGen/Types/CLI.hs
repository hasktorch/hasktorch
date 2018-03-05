{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleContexts #-}
module CodeGen.Types.CLI
  ( LibType(..)
  , describe
  , supported
  , outDir
  , outModule
  , srcDir
  , CodeGenType(..)
  , generatable
  , TemplateType(..)
  , generatedTypes
  ) where

import Data.List (intercalate)
import Data.Char (toLower)
import GHC.Generics (Generic)
import Data.Hashable (Hashable)
import qualified Data.HashSet as HS


-- | All possible libraries that we intend to support (these are all src
-- libraries in ATen)
data LibType
  = ATen
  | TH
  | THC
  | THCS
  | THCUNN
  | THNN
  | THS
  deriving (Eq, Ord, Show, Enum, Bounded, Read, Generic, Hashable)


-- | Short descriptions of each library we intend to support.
describe :: LibType -> String
describe = \case
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
supported lt = lt `HS.member` HS.fromList [TH, THC]


-- | Where generated code will be placed.
outDir :: LibType -> CodeGenType -> FilePath
outDir lt cgt = intercalate ""
  [ "output/raw/"
  , toLowers lt ++ "/"
  , "src/"
  , if cgt == GenericFiles then "generic/" else ""
  , "Torch/FFI/" ++ show lt
  ]
 where
  toLowers :: Show a => a -> String
  toLowers = map toLower . show


-- | The prefix of the output module name
outModule :: LibType -> String
outModule lt = "Torch.FFI." ++ show lt


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
generatedTypes :: CodeGenType -> [TemplateType]
generatedTypes = \case
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

