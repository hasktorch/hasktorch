{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Unmanaged.Type.SparseTensorRef where

import qualified Data.Map as Map
import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types as C
import Torch.Internal.Class
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<ATen/ATen.h>"

C.include "<vector>"

newSparseTensorRef_t ::
  Ptr Tensor ->
  IO (Ptr SparseTensorRef)
newSparseTensorRef_t _x =
  [C.throwBlock| at::SparseTensorRef* { return new at::SparseTensorRef(
    *$(at::Tensor* _x));
  }|]

deleteSparseTensorRef :: Ptr SparseTensorRef -> IO ()
deleteSparseTensorRef object = [C.throwBlock| void { delete $(at::SparseTensorRef* object);}|]

instance CppObject SparseTensorRef where
  fromPtr ptr = newForeignPtr ptr (deleteSparseTensorRef ptr)
