{-# LANGUAGE OverloadedStrings #-}

module CodeGenTypes (
  genericTypes,
  concreteTypes,

  TemplateType(..),

  HModule(..),
  TypeCategory(..),

  THType(..),
  THArg(..),
  THFunction(..),
  Parser(..)
  ) where

import Data.Text
import Text.Megaparsec
import Data.Void

-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data HModule = HModule {
  modHeader       :: FilePath,
  modPrefix       :: Text,
  modTypeTemplate :: TemplateType,
  modSuffix       :: Text,
  modFileSuffix   :: Text,
  modExtensions   :: [Text],
  modImports      :: [Text],
  modTypeDefs     :: [(Text, Text)],
  modBindings     :: [THFunction],
  modOutDir       :: Text,
  modIsTemplate   :: Bool
  } deriving Show

data TypeCategory = ReturnValue | FunctionParam

-- ----------------------------------------
-- Parsed types
-- ----------------------------------------

data THType =
  THVoidPtr
  | THVoid
  | THDescBuff
  -- Tensor
  | THTensorPtrPtr
  | THTensorPtr
  | THByteTensorPtr
  | THCharTensorPtr
  | THShortTensorPtr
  | THIntTensorPtr
  | THLongTensorPtr
  | THFloatTensorPtr
  | THDoubleTensorPtr
  | THHalfTensorPtr
  -- Storage
  | THStoragePtr
  | THByteStoragePtr
  | THCharStoragePtr
  | THShortStoragePtr
  | THIntStoragePtr
  | THLongStoragePtr
  | THFloatStoragePtr
  | THDoubleStoragePtr
  | THHalfStoragePtr
  -- Other
  | THGeneratorPtr
  | THAllocatorPtr
  | THPtrDiff
  -- Primitive
  | THFloatPtr
  | THFloat
  | THDoublePtr
  | THDouble
  | THLongPtrPtr
  | THLongPtr
  | THLong
  | THIntPtr
  | THInt
  | THUInt64
  | THUInt64Ptr
  | THUInt64PtrPtr
  | THInt64
  | THInt64Ptr
  | THInt64PtrPtr
  | THSize
  | THCharPtrPtr
  | THCharPtr
  | THChar
  | THShortPtr
  | THShort
  | THHalfPtr
  | THHalf
  | THFilePtr
  -- Templates
  | THRealPtr
  | THReal
  | THAccRealPtr
  | THAccReal
  deriving (Eq, Show)

data THArg = THArg {
  thArgType :: THType,
  thArgName :: Text
  } deriving (Eq, Show)

data THFunction = THFunction {
                funName :: Text,
                funArgs :: [THArg],
                funReturn :: THType
                } deriving (Eq, Show)

type Parser = Parsec Void String

-- ----------------------------------------
-- Types for representing templating
-- ----------------------------------------

data TemplateType = GenByte
                  | GenChar
                  | GenDouble
                  | GenFloat
                  | GenHalf
                  | GenInt
                  | GenLong
                  | GenShort
                  | GenNothing deriving (Eq, Ord, Show)

-- List used to iterate through all template types
genericTypes :: [TemplateType]
genericTypes = [GenByte, GenChar,
                GenDouble, GenFloat, GenHalf,
                GenInt, GenLong, GenShort]

concreteTypes :: [TemplateType]
concreteTypes = [GenNothing]
