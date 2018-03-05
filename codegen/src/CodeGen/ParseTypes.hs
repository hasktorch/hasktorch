{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module CodeGen.ParseTypes
  ( genericTypes
  , concreteTypes
  , TemplateType(..)

  , HModule(..)
  , ModuleSuffix(..)
  , FileSuffix(..)
  , TextPath(..)
  , IsTemplate(..)
  , TypeCategory(..)

  , THType(..)
  , THArg(..)
  , THFunction(..)
  , Parser(..)
  ) where

import CodeGen.Prelude
import CodeGen.CLITypes

-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data HModule = HModule
  { modPrefix       :: LibType
  , modExtensions   :: [Text]
  , modImports      :: [Text]
  , modTypeDefs     :: [(Text, Text)]
  , modHeader       :: FilePath
  , modTypeTemplate :: TemplateType
  , modSuffix       :: ModuleSuffix
  , modFileSuffix   :: FileSuffix
  , modBindings     :: [THFunction]
  , modOutDir       :: TextPath
  , modIsTemplate   :: IsTemplate
  } deriving Show

newtype ModuleSuffix = ModuleSuffix { textSuffix :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

newtype FileSuffix = FileSuffix { textFileSuffix :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

newtype TextPath = TextPath { textPath :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

newtype IsTemplate = IsTemplate Bool
  deriving newtype (Bounded, Enum, Eq, Ord, Read, Show)

data TypeCategory
  = ReturnValue
  | FunctionParam

-- ----------------------------------------
-- Parsed types
-- ----------------------------------------

newtype APtr x = APtr x

data THType
  = THVoidPtr
  | THBool
  | THVoid
  | THDescBuff
  -- NN
  | THNNStatePtr
  | THIndexTensorPtr
  | THIntegerTensorPtr
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
  | THUInt32
  | THUInt32Ptr
  | THUInt32PtrPtr
  | THUInt16
  | THUInt16Ptr
  | THUInt16PtrPtr
  | THUInt8
  | THUInt8Ptr
  | THUInt8PtrPtr

  | THInt64
  | THInt64Ptr
  | THInt64PtrPtr
  | THInt32
  | THInt32Ptr
  | THInt32PtrPtr
  | THInt16
  | THInt16Ptr
  | THInt16PtrPtr
  | THInt8
  | THInt8Ptr
  | THInt8PtrPtr

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


data THArg = THArg
  { thArgType :: THType
  , thArgName :: Text
  } deriving (Eq, Show)

data THFunction = THFunction
  { funName :: Text
  , funArgs :: [THArg]
  , funReturn :: THType
  } deriving (Eq, Show)

type Parser = Parsec Void String

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
genericTypes :: [TemplateType]
genericTypes =
  [ GenByte
  , GenChar
  , GenDouble
  , GenFloat
  , GenHalf
  , GenInt
  , GenLong
  , GenShort
  ]

concreteTypes :: [TemplateType]
concreteTypes = [GenNothing]
