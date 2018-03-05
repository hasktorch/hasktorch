{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
module CodeGen.Types.Parsed where

import CodeGen.Prelude

-- ----------------------------------------
-- Parsed types
-- ----------------------------------------

-- NN
data THNNType
  = THNNStatePtr
  | THIndexTensorPtr
  | THIntegerTensorPtr

data THType
  = APtr THType

  | THBool
  | THVoid
  | THDescBuff

  -- Tensor
  | THTensor
  | THByteTensor
  | THCharTensor
  | THShortTensor
  | THIntTensor
  | THLongTensor
  | THFloatTensor
  | THDoubleTensor
  | THHalfTensor
  -- Storage
  | THStorage
  | THByteStorage
  | THCharStorage
  | THShortStorage
  | THIntStorage
  | THLongStorage
  | THFloatStorage
  | THDoubleStorage
  | THHalfStorage
  -- Other
  | THGenerator
  | THAllocator
  | THPtrDiff
  | THFile

  -- Primitive
  | THFloat
  | THDouble
  | THLong
  | THInt

  | THUInt64
  | THUInt32
  | THUInt16
  | THUInt8

  | THInt64
  | THInt32
  | THInt16
  | THInt8

  | THSize
  | THChar
  | THShort
  | THHalfPtr
  | THHalf

  -- Templates
  | THReal
  | THAccReal
  deriving (Eq, Show, Generic, Hashable)


data THArg = THArg
  { thArgType :: THType
  , thArgName :: Text
  } deriving (Eq, Show, Generic, Hashable)

data THFunction = THFunction
  { funName :: Text
  , funArgs :: [THArg]
  , funReturn :: THType
  } deriving (Eq, Show, Generic, Hashable)

type Parser = Parsec Void String


