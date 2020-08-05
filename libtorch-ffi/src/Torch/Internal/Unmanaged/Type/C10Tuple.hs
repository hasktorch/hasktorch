{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Internal.Unmanaged.Type.C10Tuple where


import qualified Data.Map                         as Map
import           Foreign
import           Foreign.C.String
import           Foreign.C.Types
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C
import           Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/core/ivalue.h>"
C.include "<vector>"



newC10Tuple
  :: IO (Ptr (C10Ptr IVTuple))
newC10Tuple  =
  [C.throwBlock| c10::intrusive_ptr<at::ivalue::Tuple>* { return new c10::intrusive_ptr<at::ivalue::Tuple>(
    );
  }|]



c10Tuple_empty
  :: Ptr (C10Ptr IVTuple)
  -> IO (CBool)
c10Tuple_empty _obj =
  [C.throwBlock| bool { return (*$(c10::intrusive_ptr<at::ivalue::Tuple>* _obj))->elements().empty(
    );
  }|]

c10Tuple_size
  :: Ptr (C10Ptr IVTuple)
  -> IO (CSize)
c10Tuple_size _obj =
  [C.throwBlock| size_t { return (*$(c10::intrusive_ptr<at::ivalue::Tuple>* _obj))->elements().size(
    );
  }|]

c10Tuple_at
  :: Ptr (C10Ptr IVTuple)
  -> CSize
  -> IO (Ptr IValue)
c10Tuple_at _obj _s =
  [C.throwBlock| at::IValue* { return new at::IValue((*$(c10::intrusive_ptr<at::ivalue::Tuple>* _obj))->elements()[$(size_t _s)]);
  }|]

c10Tuple_push_back
  :: Ptr (C10Ptr IVTuple)
  -> Ptr IValue
  -> IO (())
c10Tuple_push_back _obj _v =
  [C.throwBlock| void {  (*$(c10::intrusive_ptr<at::ivalue::Tuple>* _obj))->elements().push_back(
    *$(at::IValue* _v));
  }|]



