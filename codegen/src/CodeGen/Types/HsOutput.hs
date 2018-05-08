{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module CodeGen.Types.HsOutput
  ( HModule(..)
  , ModuleSuffix(..)
  , FileSuffix(..)
  , TextPath(..)
  , makeModule

  , TypeCategory(..)

  , FunctionName(..)

  , CRep(..)
  , HsRep(..)
  , stripModule

  , CTensor(..)
  , CReal(..)
  , CAccReal(..)
  , CStorage(..)

  , HasAlias(..)
  ) where

import CodeGen.Prelude
import CodeGen.Types.CLI
import CodeGen.Types.Parsed
import qualified Data.Text as T



-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data HModule = HModule
  { lib          :: LibType
  , extensions   :: [Text]
  , imports      :: [Text]
  , typeDefs     :: [(Text, Text)]
  , header       :: FilePath
  , typeTemplate :: TemplateType
  , suffix       :: ModuleSuffix
  , fileSuffix   :: FileSuffix
  , bindings     :: [Function]
  , modOutDir    :: TextPath
  , isTemplate   :: CodeGenType
  } deriving Show

newtype ModuleSuffix = ModuleSuffix { textSuffix :: Text }
  deriving newtype (IsString, Semigroup, Monoid, Ord, Read, Eq, Show)

newtype FileSuffix = FileSuffix { textFileSuffix :: Text }
  deriving newtype (IsString, Semigroup, Monoid, Ord, Read, Eq, Show)

newtype TextPath = TextPath { textPath :: Text }
  deriving newtype (IsString, Semigroup, Monoid, Ord, Read, Eq, Show)

makeModule
  :: LibType
  -> TextPath
  -> CodeGenType
  -> FilePath
  -> ModuleSuffix
  -> FileSuffix
  -> TemplateType
  -> [Function]
  -> HModule
makeModule lt a0 a1 a2 a3 a4 a5 a6
  = HModule
  { lib = lt
  , extensions = ["ForeignFunctionInterface"]
  , imports = ["Foreign", "Foreign.C.Types", "Data.Word", "Data.Int"] <> torchtypes
  , typeDefs = []
  , modOutDir = a0
  , isTemplate = a1
  , header = a2
  , suffix = a3
  , fileSuffix = a4
  , typeTemplate = a5
  , bindings = a6
  }
 where
  torchtypes :: [Text]
  torchtypes = case lt of
    THC    -> go [TH, THC]
    THCUNN -> go [TH, THC]
    THNN   -> go [TH]
    rest   -> go [rest]
    where
      go :: [LibType] -> [Text]
      go ls = (("Torch.Types." <>) . tshow) <$> ls


data TypeCategory
  = ReturnValue
  | FunctionParam

-------------------------------------------------------------------------------

-- | a concrete type for function names
newtype FunctionName = FunctionName Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

newtype CRep = CRep Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

newtype HsRep = HsRep Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

stripModule :: HsRep -> Text
stripModule (HsRep t) = T.takeWhileEnd (/= '.') t

data CTensor  = CTensor  HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CReal    = CReal    HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CAccReal = CAccReal HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CStorage = CStorage HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

-- ========================================================================= --

class HasAlias t where
  alias :: t -> Text

instance HasAlias CTensor  where alias (CTensor t _)  = _alias "CTensor" t
instance HasAlias CReal    where alias (CReal t _)    = _alias "CReal" t
instance HasAlias CAccReal where alias (CAccReal t _) = _alias "CAccReal" t
instance HasAlias CStorage where alias (CStorage t _) = _alias "CStorage" t

_alias :: Text -> HsRep -> Text
_alias a (HsRep t) = T.intercalate " " ["type", a, "=", t]


