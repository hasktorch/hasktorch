
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.C10List where


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

C.include "<ATen/core/List.h>"
C.include "<ATen/core/ivalue.h>"
C.include "<vector>"

newC10ListIValue :: Ptr IValue -> IO (Ptr (C10List IValue))
newC10ListIValue elem = [C.throwBlock| c10::List<at::IValue>* { return new c10::impl::GenericList($(at::IValue* elem)->type()); }|]

newC10ListTensor :: IO (Ptr (C10List Tensor))
newC10ListTensor = [C.throwBlock| c10::List<at::Tensor>* { return new c10::List<at::Tensor>(); }|]

newC10ListOptionalTensor :: IO (Ptr (C10List (C10Optional Tensor)))
newC10ListOptionalTensor = [C.throwBlock| c10::List<c10::optional<at::Tensor>>* { return new c10::List<c10::optional<at::Tensor>>(); }|]

newC10ListDouble :: IO (Ptr (C10List CDouble))
newC10ListDouble = [C.throwBlock| c10::List<double>* { return new c10::List<double>(); }|]

newC10ListInt :: IO (Ptr (C10List Int64))
newC10ListInt = [C.throwBlock| c10::List<int64_t>* { return new c10::List<int64_t>(); }|]

newC10ListBool :: IO (Ptr (C10List CBool))
newC10ListBool = [C.throwBlock| c10::List<bool>* { return new c10::List<bool>(); }|]




c10ListIValue_empty :: Ptr (C10List IValue) -> IO (CBool)
c10ListIValue_empty _obj = [C.throwBlock| bool { return (*$(c10::List<at::IValue>* _obj)).empty(); }|]

c10ListTensor_empty :: Ptr (C10List Tensor) -> IO (CBool)
c10ListTensor_empty _obj = [C.throwBlock| bool { return (*$(c10::List<at::Tensor>* _obj)).empty(); }|]

c10ListOptionalTensor_empty :: Ptr (C10List (C10Optional Tensor)) -> IO (CBool)
c10ListOptionalTensor_empty _obj = [C.throwBlock| bool { return (*$(c10::List<c10::optional<at::Tensor>>* _obj)).empty(); }|]

c10ListDouble_empty :: Ptr (C10List CDouble) -> IO (CBool)
c10ListDouble_empty _obj = [C.throwBlock| bool { return (*$(c10::List<double>* _obj)).empty(); }|]

c10ListInt_empty :: Ptr (C10List Int64) -> IO (CBool)
c10ListInt_empty _obj = [C.throwBlock| bool { return (*$(c10::List<int64_t>* _obj)).empty(); }|]

c10ListBool_empty :: Ptr (C10List CBool) -> IO (CBool)
c10ListBool_empty _obj = [C.throwBlock| bool { return (*$(c10::List<bool>* _obj)).empty(); }|]

c10ListIValue_size :: Ptr (C10List IValue) -> IO (CSize)
c10ListIValue_size _obj = [C.throwBlock| size_t { return (*$(c10::List<at::IValue>* _obj)).size(); }|]

c10ListTensor_size :: Ptr (C10List Tensor) -> IO (CSize)
c10ListTensor_size _obj = [C.throwBlock| size_t { return (*$(c10::List<at::Tensor>* _obj)).size(); }|]

c10ListOptionalTensor_size :: Ptr (C10List (C10Optional Tensor)) -> IO (CSize)
c10ListOptionalTensor_size _obj = [C.throwBlock| size_t { return (*$(c10::List<c10::optional<at::Tensor>>* _obj)).size(); }|]

c10ListDouble_size :: Ptr (C10List CDouble) -> IO (CSize)
c10ListDouble_size _obj = [C.throwBlock| size_t { return (*$(c10::List<double>* _obj)).size(); }|]

c10ListInt_size :: Ptr (C10List Int64) -> IO (CSize)
c10ListInt_size _obj = [C.throwBlock| size_t { return (*$(c10::List<int64_t>* _obj)).size(); }|]

c10ListBool_size :: Ptr (C10List CBool) -> IO (CSize)
c10ListBool_size _obj = [C.throwBlock| size_t { return (*$(c10::List<bool>* _obj)).size(); }|]

c10ListIValue_at :: Ptr (C10List IValue) -> CSize -> IO (Ptr IValue)
c10ListIValue_at _obj _s = [C.throwBlock| at::IValue* { return new at::IValue((*$(c10::List<at::IValue>* _obj))[$(size_t _s)]); }|]

c10ListTensor_at :: Ptr (C10List Tensor) -> CSize -> IO (Ptr Tensor)
c10ListTensor_at _obj _s = [C.throwBlock| at::Tensor* { return new at::Tensor((*$(c10::List<at::Tensor>* _obj))[$(size_t _s)]); }|]

c10ListOptionalTensor_at :: Ptr (C10List (C10Optional Tensor)) -> CSize -> IO (Ptr Tensor)
c10ListOptionalTensor_at _obj _s = [C.throwBlock| at::Tensor* {
    c10::List<c10::optional<at::Tensor>>& list = *$(c10::List<c10::optional<at::Tensor>>* _obj);
    c10::optional<at::Tensor> v = list[$(size_t _s)];
    return new at::Tensor(v.value());
  }|]

c10ListDouble_at :: Ptr (C10List CDouble) -> CSize -> IO CDouble
c10ListDouble_at _obj _s = [C.throwBlock| double { return ((*$(c10::List<double>* _obj))[$(size_t _s)]); }|]

c10ListInt_at :: Ptr (C10List Int64) -> CSize -> IO Int64
c10ListInt_at _obj _s = [C.throwBlock| int64_t { return (int64_t)((*$(c10::List<int64_t>* _obj))[$(size_t _s)]); }|]

c10ListBool_at :: Ptr (C10List CBool) -> CSize -> IO CBool
c10ListBool_at _obj _s = [C.throwBlock| bool { return ((*$(c10::List<bool>* _obj))[$(size_t _s)]); }|]

c10ListIValue_push_back :: Ptr (C10List IValue) -> Ptr IValue -> IO ()
c10ListIValue_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<at::IValue>* _obj)).push_back(*$(at::IValue* _v)); }|]

c10ListTensor_push_back :: Ptr (C10List Tensor) -> Ptr Tensor -> IO ()
c10ListTensor_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<at::Tensor>* _obj)).push_back(*$(at::Tensor* _v)); }|]

c10ListOptionalTensor_push_back :: Ptr (C10List (C10Optional Tensor)) -> Ptr Tensor -> IO ()
c10ListOptionalTensor_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<c10::optional<at::Tensor>>* _obj)).push_back(*$(at::Tensor* _v)); }|]

c10ListDouble_push_back :: Ptr (C10List CDouble) -> CDouble -> IO ()
c10ListDouble_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<double>* _obj)).push_back($(double _v)); }|]

c10ListInt_push_back :: Ptr (C10List Int64) -> Int64 -> IO ()
c10ListInt_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<int64_t>* _obj)).push_back($(int64_t _v)); }|]

c10ListBool_push_back :: Ptr (C10List CBool) -> CBool -> IO ()
c10ListBool_push_back _obj _v = [C.throwBlock| void {  (*$(c10::List<bool>* _obj)).push_back($(bool _v)); }|]



