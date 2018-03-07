{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
module CodeGen.Types.Parsed where

import CodeGen.Prelude

-- ----------------------------------------
-- Parsed types
-- ----------------------------------------
data Parsable
  = Ptr Parsable
  | TenType TenType
  | NNType NNType
  | CType CType
  deriving (Eq, Show, Generic, Hashable)

data CType
  = CBool
  | CVoid
  | CPtrdiff
  | CFloat
  | CDouble
  | CLong

  | CUInt64
  | CUInt32
  | CUInt16
  | CUInt8

  | CInt64
  | CInt32
  | CInt16
  | CInt8

  | CInt -- must come _after_ all the other int types

  | CSize
  | CChar
  | CShort
  deriving (Eq, Show, Generic, Hashable, Bounded, Enum)

data TenType
  = Tensor
  | ByteTensor
  | CharTensor
  | ShortTensor
  | IntTensor
  | LongTensor
  | FloatTensor
  | DoubleTensor
  | HalfTensor

  | Storage
  | ByteStorage
  | CharStorage
  | ShortStorage
  | IntStorage
  | LongStorage
  | FloatStorage
  | DoubleStorage
  | HalfStorage

  -- Other
  | DescBuff
  | Generator
  | Allocator
  | File
  | Half
  | Real
  | AccReal
  deriving (Eq, Show, Generic, Hashable, Bounded, Enum)


data NNType
  = NNState
  | IndexTensor
  | IntegerTensor
  deriving (Eq, Show, Generic, Hashable, Bounded, Enum)


data Arg = Arg
  { argType :: Parsable
  , argName :: Text
  } deriving (Eq, Show, Generic, Hashable)

data Function = Function
  { funName :: Text
  , funArgs :: [Arg]
  , funReturn :: Parsable
  } deriving (Eq, Show, Generic, Hashable)

type Parser = Parsec Void String


