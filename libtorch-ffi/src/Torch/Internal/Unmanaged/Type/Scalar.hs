{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Unmanaged.Type.Scalar where

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

newScalar ::
  IO (Ptr Scalar)
newScalar =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    );
  }|]

newScalar_i ::
  CInt ->
  IO (Ptr Scalar)
newScalar_i _a =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    $(int _a));
  }|]

newScalar_d ::
  CDouble ->
  IO (Ptr Scalar)
newScalar_d _a =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    $(double _a));
  }|]

deleteScalar :: Ptr Scalar -> IO ()
deleteScalar object = [C.throwBlock| void { delete $(at::Scalar* object);}|]

instance CppObject Scalar where
  fromPtr ptr = newForeignPtr ptr (deleteScalar ptr)
