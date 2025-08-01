{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Unmanaged.Type.StdOptional where

import Foreign
import Foreign.C.Types
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<ATen/Tensor.h>"
C.include "<optional>"

stdOptionalTensor_create :: Ptr Tensor -> IO (Ptr (StdOptional Tensor))
stdOptionalTensor_create tensor =
  [C.throwBlock| std::optional<at::Tensor>* {
    return new std::optional<at::Tensor>(std::make_optional(*$(at::Tensor* tensor)));
}|]

stdOptionalTensor_empty :: IO (Ptr (StdOptional Tensor))
stdOptionalTensor_empty =
  [C.throwBlock| std::optional<at::Tensor>* {
    return new std::optional<at::Tensor>(std::nullopt);
}|]

stdOptionalTensor_has_value :: Ptr (StdOptional Tensor) -> IO CBool
stdOptionalTensor_has_value _obj =
  [C.throwBlock| bool {
    return $(std::optional<at::Tensor>* _obj)->has_value();
}|]

stdOptionalTensor_value :: Ptr (StdOptional Tensor) -> IO (Ptr Tensor)
stdOptionalTensor_value _obj =
  [C.throwBlock| at::Tensor* {
    return new at::Tensor($(std::optional<at::Tensor>* _obj)->value());
}|]
