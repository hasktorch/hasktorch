
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Scalar where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/Scalar.h>"
C.include "<vector>"



newScalar
  :: IO (Ptr Scalar)
newScalar  =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    );
  }|]

newScalar_i
  :: CInt
  -> IO (Ptr Scalar)
newScalar_i _a =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    $(int _a));
  }|]

newScalar_d
  :: CDouble
  -> IO (Ptr Scalar)
newScalar_d _a =
  [C.throwBlock| at::Scalar* { return new at::Scalar(
    $(double _a));
  }|]



