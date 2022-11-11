
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.TensorList where


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



C.include "<ATen/Tensor.h>"
C.include "<vector>"



newTensorList
  :: IO (Ptr TensorList)
newTensorList  =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(
    );
  }|]

tensorList_empty
  :: Ptr TensorList
  -> IO (CBool)
tensorList_empty _obj =
  [C.throwBlock| bool { return (*$(std::vector<at::Tensor>* _obj)).empty(
    );
  }|]

tensorList_size
  :: Ptr TensorList
  -> IO (CSize)
tensorList_size _obj =
  [C.throwBlock| size_t { return (*$(std::vector<at::Tensor>* _obj)).size(
    );
  }|]

tensorList_at_s
  :: Ptr TensorList
  -> CSize
  -> IO (Ptr Tensor)
tensorList_at_s _obj _s =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(std::vector<at::Tensor>* _obj)).at(
    $(size_t _s)));
  }|]

tensorList_push_back_t
  :: Ptr TensorList
  -> Ptr Tensor
  -> IO (())
tensorList_push_back_t _obj _v =
  [C.throwBlock| void {  (*$(std::vector<at::Tensor>* _obj)).push_back(
    *$(at::Tensor* _v));
  }|]

