{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
module CodeGen.Types.Parsed where

import CodeGen.Prelude

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


