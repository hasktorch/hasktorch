
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.Tensor where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"



newTensor
  :: IO (Ptr Tensor)
newTensor  =
  [C.block| at::Tensor* { return new at::Tensor(
    );
  }|]

new_TensorTensor
  :: Ptr Tensor
  -> IO (Ptr Tensor)
new_TensorTensor _x =
  [C.block| at::Tensor* { return new at::Tensor(
    *$(at::Tensor* _x));
  }|]



deleteTensor :: Ptr Tensor -> IO ()
deleteTensor object = [C.block| void { delete $(at::Tensor* object);}|]

instance CppObject Tensor where
  fromPtr ptr = newForeignPtr ptr (deleteTensor ptr)



tensor_dim
  :: Ptr Tensor
  -> IO (Int64)
tensor_dim _obj =
  [C.block| int64_t { return ($(at::Tensor* _obj)->dim(
    ));
  }|]

tensor_storage_offset
  :: Ptr Tensor
  -> IO (Int64)
tensor_storage_offset _obj =
  [C.block| int64_t { return ($(at::Tensor* _obj)->storage_offset(
    ));
  }|]

tensor_defined
  :: Ptr Tensor
  -> IO (CBool)
tensor_defined _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->defined(
    ));
  }|]

tensor_reset
  :: Ptr Tensor
  -> IO (())
tensor_reset _obj =
  [C.block| void {  ($(at::Tensor* _obj)->reset(
    ));
  }|]

tensor_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_use_count _obj =
  [C.block| size_t { return ($(at::Tensor* _obj)->use_count(
    ));
  }|]

tensor_weak_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_weak_use_count _obj =
  [C.block| size_t { return ($(at::Tensor* _obj)->weak_use_count(
    ));
  }|]

tensor_ndimension
  :: Ptr Tensor
  -> IO (Int64)
tensor_ndimension _obj =
  [C.block| int64_t { return ($(at::Tensor* _obj)->ndimension(
    ));
  }|]

tensor_is_contiguous
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_contiguous _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->is_contiguous(
    ));
  }|]

tensor_nbytes
  :: Ptr Tensor
  -> IO (CSize)
tensor_nbytes _obj =
  [C.block| size_t { return ($(at::Tensor* _obj)->nbytes(
    ));
  }|]

tensor_itemsize
  :: Ptr Tensor
  -> IO (CSize)
tensor_itemsize _obj =
  [C.block| size_t { return ($(at::Tensor* _obj)->itemsize(
    ));
  }|]

tensor_element_size
  :: Ptr Tensor
  -> IO (CSize)
tensor_element_size _obj =
  [C.block| size_t { return ($(at::Tensor* _obj)->element_size(
    ));
  }|]

tensor_is_variable
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_variable _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->is_variable(
    ));
  }|]

tensor_get_device
  :: Ptr Tensor
  -> IO (Int64)
tensor_get_device _obj =
  [C.block| int64_t { return ($(at::Tensor* _obj)->get_device(
    ));
  }|]

tensor_is_cuda
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_cuda _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->is_cuda(
    ));
  }|]

tensor_is_hip
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_hip _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->is_hip(
    ));
  }|]

tensor_is_sparse
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_sparse _obj =
  [C.block| bool { return ($(at::Tensor* _obj)->is_sparse(
    ));
  }|]

tensor_print
  :: Ptr Tensor
  -> IO (())
tensor_print _obj =
  [C.block| void {  ($(at::Tensor* _obj)->print(
    ));
  }|]

