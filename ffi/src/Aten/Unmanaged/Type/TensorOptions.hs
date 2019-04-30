
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.TensorOptions where


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



newTensorOptions
  :: DeviceType
  -> IO (Ptr TensorOptions)
newTensorOptions _d =
  [C.block| at::TensorOptions* { return new at::TensorOptions(
    $(at::DeviceType _d));
  }|]



deleteTensorOptions :: Ptr TensorOptions -> IO ()
deleteTensorOptions object = [C.block| void { delete $(at::TensorOptions* object);}|]

instance CppObject TensorOptions where
  fromPtr ptr = newForeignPtr ptr (deleteTensorOptions ptr)



tensorOptions_dtype
  :: Ptr TensorOptions
  -> ScalarType
  -> IO (Ptr TensorOptions)
tensorOptions_dtype _obj _t =
  [C.block| at::TensorOptions* { return new at::TensorOptions($(at::TensorOptions* _obj)->dtype(
    $(at::ScalarType _t)));
  }|]

