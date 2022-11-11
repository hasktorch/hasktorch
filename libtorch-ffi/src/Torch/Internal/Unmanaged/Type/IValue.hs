
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Torch.Internal.Unmanaged.Type.IValue where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/core/ivalue.h>"
C.include "<vector>"

class IValueLike a b where
  toIValue :: a -> IO b
  fromIValue :: b -> IO a

instance IValueLike (Ptr IValue) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(at::IValue* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| at::IValue* { return new at::IValue((*$(at::IValue* _obj)).toIValue(
      ));
    }|]

instance IValueLike (Ptr Tensor) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(at::Tensor* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::IValue* _obj)).toTensor(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr IVTuple)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<at::ivalue::Tuple>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<at::ivalue::Tuple>* { return new c10::intrusive_ptr<at::ivalue::Tuple>((*$(at::IValue* _obj)).toTuple(
      ));
    }|]

instance IValueLike (Ptr (C10Dict '(IValue,IValue))) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::Dict<at::IValue,at::IValue>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::Dict<at::IValue,at::IValue>* { return new c10::Dict<at::IValue,at::IValue>((*$(at::IValue* _obj)).toGenericDict(
      ));
    }|]

instance IValueLike (Ptr (C10List IValue)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::List<at::IValue>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::List<at::IValue>* { return new c10::List<at::IValue>((*$(at::IValue* _obj)).toList(
      ));
    }|]

instance IValueLike (Ptr (C10List Tensor)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::List<at::Tensor>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::List<at::Tensor>* { return new c10::List<at::Tensor>((*$(at::IValue* _obj)).toTensorList(
      ));
    }|]

instance IValueLike (Ptr (C10List CBool)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::List<bool>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::List<bool>* { return new c10::List<bool>((*$(at::IValue* _obj)).toBoolList(
      ));
    }|]

instance IValueLike (Ptr (C10List Int64)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::List<int64_t>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::List<int64_t>* { return new c10::List<int64_t>((*$(at::IValue* _obj)).toIntList(
      ));
    }|]

instance IValueLike (Ptr (C10List CDouble)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::List<double>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::List<double>* { return new c10::List<double>((*$(at::IValue* _obj)).toDoubleList(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr IVObject)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<at::ivalue::Object>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<at::ivalue::Object>* { return new c10::intrusive_ptr<at::ivalue::Object>((*$(at::IValue* _obj)).toObject(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr IVFuture)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<at::ivalue::Future>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<at::ivalue::Future>* { return new c10::intrusive_ptr<at::ivalue::Future>((*$(at::IValue* _obj)).toFuture(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr IVConstantString)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<at::ivalue::ConstantString>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<at::ivalue::ConstantString>* { return new c10::intrusive_ptr<at::ivalue::ConstantString>((*$(at::IValue* _obj)).toString(
      ));
    }|]

instance IValueLike (Ptr StdString) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(std::string* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| std::string* { return new std::string((*$(at::IValue* _obj)).toStringRef(
      ));
    }|]

instance IValueLike (Ptr Scalar) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(at::Scalar* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| at::Scalar* { return new at::Scalar((*$(at::IValue* _obj)).toScalar(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr Capsule)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<torch::jit::CustomClassHolder>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<torch::jit::CustomClassHolder>* { return new c10::intrusive_ptr<torch::jit::CustomClassHolder>((*$(at::IValue* _obj)).toCapsule(
      ));
    }|]

instance IValueLike (Ptr (C10Ptr Blob)) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::intrusive_ptr<caffe2::Blob>* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::intrusive_ptr<caffe2::Blob>* { return new c10::intrusive_ptr<caffe2::Blob>((*$(at::IValue* _obj)).toBlob(
      ));
    }|]

instance IValueLike CDouble (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      $(double _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| double { return ((*$(at::IValue* _obj)).toDouble(
      ));
    }|]

instance IValueLike Int64 (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      $(int64_t _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| int64_t { return ((*$(at::IValue* _obj)).toInt(
      ));
    }|]

instance IValueLike Int32 (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      $(int32_t _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| int32_t { return ((*$(at::IValue* _obj)).toInt(
      ));
    }|]

instance IValueLike CBool (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      $(bool _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| bool { return ((*$(at::IValue* _obj)).toBool(
      ));
    }|]


instance IValueLike (Ptr Device) (Ptr IValue) where
  toIValue _x =
    [C.throwBlock| at::IValue* { return new at::IValue(
      *$(c10::Device* _x));
    }|]
  fromIValue _obj = 
    [C.throwBlock| c10::Device* { return new c10::Device((*$(at::IValue* _obj)).toDevice(
      ));
    }|]

newIValue
  :: IO (Ptr IValue)
newIValue  =
  [C.throwBlock| at::IValue* { return new at::IValue() ; }|]


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

iValue_isBool
  :: Ptr IValue
  -> IO (CBool)
iValue_isBool _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isBool(
    );
  }|]

iValue_isObject
  :: Ptr IValue
  -> IO (CBool)
iValue_isObject _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isObject(
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

iValue_isList
  :: Ptr IValue
  -> IO (CBool)
iValue_isList _obj =
  [C.throwBlock| bool { return (*$(at::IValue* _obj)).isList(
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

