
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.C10ListTensor where


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



newC10ListTensor
  :: IO (Ptr (C10List Tensor))
newC10ListTensor  =
  [C.throwBlock| c10::List<at::Tensor>* { return new c10::List<at::Tensor>(
    );
  }|]



deleteC10ListTensor :: Ptr (C10List Tensor) -> IO ()
deleteC10ListTensor object = [C.throwBlock| void { delete $(c10::List<at::Tensor>* object);}|]

instance CppObject (C10List Tensor) where
  fromPtr ptr = newForeignPtr ptr (deleteC10ListTensor ptr)



c10ListTensor_empty
  :: Ptr (C10List Tensor)
  -> IO (CBool)
c10ListTensor_empty _obj =
  [C.throwBlock| bool { return (*$(c10::List<at::Tensor>* _obj)).empty(
    );
  }|]

c10ListTensor_size
  :: Ptr (C10List Tensor)
  -> IO (CSize)
c10ListTensor_size _obj =
  [C.throwBlock| size_t { return (*$(c10::List<at::Tensor>* _obj)).size(
    );
  }|]

c10ListTensor_at_s
  :: Ptr (C10List Tensor)
  -> CSize
  -> IO (Ptr Tensor)
c10ListTensor_at_s _obj _s =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(c10::List<at::Tensor>* _obj))[$(size_t _s)]);
  }|]

c10ListTensor_push_back_t
  :: Ptr (C10List Tensor)
  -> Ptr Tensor
  -> IO (())
c10ListTensor_push_back_t _obj _v =
  [C.throwBlock| void {  (*$(c10::List<at::Tensor>* _obj)).push_back(
    *$(at::Tensor* _v));
  }|]



