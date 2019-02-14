{-

select compiler using:
`stack build --ghc-options='-pgmc [gcc/clang path]'`

run in ghci using:
`stack ghci --ghc-options='-fobject-code' --main-is ffi-experimental:exe:cpp-test`

-}

{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module NativeFunctions where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

import Foreign.C.String
import Foreign.C.Types
import Foreign

data Tensor

C.context $ C.cppCtx <> mempty {
    C.ctxTypesTable = Map.fromList [
      (C.TypeName "at::Tensor", [t|Tensor|])
    ]
}

C.include "<ATen/ATen.h>"

cpp_cast_Byte :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
cpp_cast_Byte a b =  [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Byte(*$(at::Tensor* a), $(bool b))); }|]
