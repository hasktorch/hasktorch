
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Symbol where


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



newSymbol
  :: IO (Ptr Symbol)
newSymbol  =
  [C.throwBlock| at::Symbol* { return new at::Symbol(
    );
  }|]



deleteSymbol :: Ptr Symbol -> IO ()
deleteSymbol object = [C.throwBlock| void { delete $(at::Symbol* object);}|]

instance CppObject Symbol where
  fromPtr ptr = newForeignPtr ptr (deleteSymbol ptr)



symbol_is_attr
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_attr _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_attr(
    );
  }|]

symbol_is_aten
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_aten _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_aten(
    );
  }|]

symbol_is_prim
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_prim _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_prim(
    );
  }|]

symbol_is_onnx
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_onnx _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_onnx(
    );
  }|]

symbol_is_user
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_user _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_user(
    );
  }|]

symbol_is_caffe2
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_caffe2 _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_caffe2(
    );
  }|]

symbol_is_dimname
  :: Ptr Symbol
  -> IO (CBool)
symbol_is_dimname _obj =
  [C.throwBlock| bool { return (*$(at::Symbol* _obj)).is_dimname(
    );
  }|]



attr_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
attr_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::attr(
    *$(std::string* _s)));
  }|]

aten_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
aten_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::aten(
    *$(std::string* _s)));
  }|]

onnx_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
onnx_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::onnx(
    *$(std::string* _s)));
  }|]

prim_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
prim_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::prim(
    *$(std::string* _s)));
  }|]

user_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
user_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::user(
    *$(std::string* _s)));
  }|]

caffe2_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
caffe2_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::caffe2(
    *$(std::string* _s)));
  }|]

dimname_s
  :: Ptr StdString
  -> IO (Ptr Symbol)
dimname_s _s =
  [C.throwBlock| at::Symbol* { return new at::Symbol(at::Symbol::dimname(
    *$(std::string* _s)));
  }|]

