
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.Scalar where


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



newScalar
  :: IO (Ptr Scalar)
newScalar  =
  [C.block| at::Scalar* { return new at::Scalar(
    );
  }|]

new_intScalar
  :: CInt
  -> IO (Ptr Scalar)
new_intScalar _a =
  [C.block| at::Scalar* { return new at::Scalar(
    $(int _a));
  }|]

new_doubleScalar
  :: CDouble
  -> IO (Ptr Scalar)
new_doubleScalar _a =
  [C.block| at::Scalar* { return new at::Scalar(
    $(double _a));
  }|]



deleteScalar :: Ptr Scalar -> IO ()
deleteScalar object = [C.block| void { delete $(at::Scalar* object);}|]

instance CppObject Scalar where
  fromPtr ptr = newForeignPtr ptr (deleteScalar ptr)



