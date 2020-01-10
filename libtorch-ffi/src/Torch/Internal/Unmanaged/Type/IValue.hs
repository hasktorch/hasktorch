
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.IValue where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"



newIValue_V
  :: Ptr IValue
  -> IO (Ptr IValue)
newIValue_V _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    *$(at::IValue* _x));
  }|]

newIValue_t
  :: Ptr Tensor
  -> IO (Ptr IValue)
newIValue_t _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    *$(at::Tensor* _x));
  }|]

newIValue_l
  :: Ptr TensorList
  -> IO (Ptr IValue)
newIValue_l _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    *$(std::vector<at::Tensor>* _x));
  }|]

newIValue_s
  :: Ptr Scalar
  -> IO (Ptr IValue)
newIValue_s _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    *$(at::Scalar* _x));
  }|]

newIValue_d
  :: CDouble
  -> IO (Ptr IValue)
newIValue_d _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    $(double _x));
  }|]

newIValue_l
  :: Int64
  -> IO (Ptr IValue)
newIValue_l _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    $(int64_t _x));
  }|]

newIValue_i
  :: Int32
  -> IO (Ptr IValue)
newIValue_i _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    $(int32_t _x));
  }|]

newIValue_b
  :: CBool
  -> IO (Ptr IValue)
newIValue_b _x =
  [C.throwBlock| at::IValue* { return new at::IValue(
    $(bool _x));
  }|]



deleteIValue :: Ptr IValue -> IO ()
deleteIValue object = [C.throwBlock| void { delete $(at::IValue* object);}|]

instance CppObject IValue where
  fromPtr ptr = newForeignPtr ptr (deleteIValue ptr)



iValue_isAliasOf_V
  :: Ptr IValue
  -> Ptr IValue
  -> IO (CBool)
iValue_isAliasOf_V _obj _rhs =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isAliasOf(
    *$(at::IValue* _rhs));
  }|]

iValue_use_count
  :: Ptr IValue
  -> IO (CSize)
iValue_use_count _obj =
  [C.throwBlock| size_t { return (*$(at::IValue* _obj)).use_count(
    );
  }|]

iValue_swap_V
  :: Ptr IValue
  -> Ptr IValue
  -> IO (())
iValue_swap_V _obj _rhs =
  [C.throwBlock| void {  (*$(at::IValue* _obj)).swap(
    *$(at::IValue* _rhs));
  }|]

iValue_isTensor
  :: Ptr IValue
  -> IO (CBool)
iValue_isTensor _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isTensor(
    );
  }|]

iValue_toIValue
  :: Ptr IValue
  -> IO (Ptr IValue)
iValue_toIValue _obj =
  [C.throwBlock| at::IValue* { return new at::IValue((*$(at::IValue* _obj)).toIValue(
    ));
  }|]

iValue_isBlob
  :: Ptr IValue
  -> IO (CBool)
iValue_isBlob _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isBlob(
    );
  }|]

iValue_isCapsule
  :: Ptr IValue
  -> IO (CBool)
iValue_isCapsule _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isCapsule(
    );
  }|]

iValue_isTuple
  :: Ptr IValue
  -> IO (CBool)
iValue_isTuple _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isTuple(
    );
  }|]

iValue_isDouble
  :: Ptr IValue
  -> IO (CBool)
iValue_isDouble _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isDouble(
    );
  }|]

iValue_toDouble
  :: Ptr IValue
  -> IO (CDouble)
iValue_toDouble _obj =
  [C.throwBlock| double { return (*$(at::IValue* _obj)).toDouble(
    );
  }|]

iValue_isFuture
  :: Ptr IValue
  -> IO (CBool)
iValue_isFuture _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isFuture(
    );
  }|]

iValue_isInt
  :: Ptr IValue
  -> IO (CBool)
iValue_isInt _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isInt(
    );
  }|]

iValue_toInt
  :: Ptr IValue
  -> IO (Int64)
iValue_toInt _obj =
  [C.throwBlock| int64_t { return (*$(at::IValue* _obj)).toInt(
    );
  }|]

iValue_isIntList
  :: Ptr IValue
  -> IO (CBool)
iValue_isIntList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isIntList(
    );
  }|]

iValue_isString
  :: Ptr IValue
  -> IO (CBool)
iValue_isString _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isString(
    );
  }|]

iValue_toStringRef
  :: Ptr IValue
  -> IO (Ptr StdString)
iValue_toStringRef _obj =
  [C.throwBlock| std::string* { return new std::string((*$(at::IValue* _obj)).toStringRef(
    ));
  }|]

iValue_isDoubleList
  :: Ptr IValue
  -> IO (CBool)
iValue_isDoubleList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isDoubleList(
    );
  }|]

iValue_isBoolList
  :: Ptr IValue
  -> IO (CBool)
iValue_isBoolList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isBoolList(
    );
  }|]

iValue_isTensorList
  :: Ptr IValue
  -> IO (CBool)
iValue_isTensorList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isTensorList(
    );
  }|]

iValue_isGenericList
  :: Ptr IValue
  -> IO (CBool)
iValue_isGenericList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isGenericList(
    );
  }|]

iValue_isGenericDict
  :: Ptr IValue
  -> IO (CBool)
iValue_isGenericDict _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isGenericDict(
    );
  }|]

iValue_isNone
  :: Ptr IValue
  -> IO (CBool)
iValue_isNone _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isNone(
    );
  }|]

iValue_toNone
  :: Ptr IValue
  -> IO (Ptr StdString)
iValue_toNone _obj =
  [C.throwBlock| std::string* { return new std::string((*$(at::IValue* _obj)).toNone(
    ));
  }|]

iValue_isScalar
  :: Ptr IValue
  -> IO (CBool)
iValue_isScalar _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isScalar(
    );
  }|]

iValue_toScalar
  :: Ptr IValue
  -> IO (Ptr Scalar)
iValue_toScalar _obj =
  [C.throwBlock| at::Scalar* { return new at::Scalar((*$(at::IValue* _obj)).toScalar(
    ));
  }|]

iValue_isDevice
  :: Ptr IValue
  -> IO (CBool)
iValue_isDevice _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isDevice(
    );
  }|]

iValue_toScalarType
  :: Ptr IValue
  -> IO (ScalarType)
iValue_toScalarType _obj =
  [C.throwBlock| at::ScalarType { return (*$(at::IValue* _obj)).toScalarType(
    );
  }|]

iValue_toLayout
  :: Ptr IValue
  -> IO (Layout)
iValue_toLayout _obj =
  [C.throwBlock| at::Layout { return (*$(at::IValue* _obj)).toLayout(
    );
  }|]

iValue_toMemoryFormat
  :: Ptr IValue
  -> IO (MemoryFormat)
iValue_toMemoryFormat _obj =
  [C.throwBlock| at::MemoryFormat { return (*$(at::IValue* _obj)).toMemoryFormat(
    );
  }|]

iValue_toQScheme
  :: Ptr IValue
  -> IO (QScheme)
iValue_toQScheme _obj =
  [C.throwBlock| at::QScheme { return (*$(at::IValue* _obj)).toQScheme(
    );
  }|]

iValue_tagKind
  :: Ptr IValue
  -> IO (Ptr StdString)
iValue_tagKind _obj =
  [C.throwBlock| std::string* { return new std::string((*$(at::IValue* _obj)).tagKind(
    ));
  }|]

iValue_isSameIdentity_V
  :: Ptr IValue
  -> Ptr IValue
  -> IO (CBool)
iValue_isSameIdentity_V _obj _rhs =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isSameIdentity(
    *$(at::IValue* _rhs));
  }|]

iValue_isPtrType
  :: Ptr IValue
  -> IO (CBool)
iValue_isPtrType _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isPtrType(
    );
  }|]

iValue_internalToPointer
  :: Ptr IValue
  -> IO (Ptr ())
iValue_internalToPointer _obj =
  [C.throwBlock| void * { return (*$(at::IValue* _obj)).internalToPointer(
    );
  }|]

iValue_clearToNone
  :: Ptr IValue
  -> IO (())
iValue_clearToNone _obj =
  [C.throwBlock| void {  (*$(at::IValue* _obj)).clearToNone(
    );
  }|]



