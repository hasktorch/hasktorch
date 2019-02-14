{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderNativeFunctions where

import Data.Yaml

import qualified Data.Yaml as Y
import Text.Megaparsec (parse, ParseErrorBundle, errorBundlePretty)
import Control.Monad (forM_)
import Text.Shakespeare.Text (st)
--import qualified Data.Text as T
import Data.Text (Text)
import qualified Data.Text.IO as T

import ParseNativeFunctions
import ParseFunctionSig

bra :: Text
bra = "["

cket :: Text
cket = "]"

decodeAndCodeGen :: String -> IO ()
decodeAndCodeGen fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction'])
  case funcs of
    Left err' -> print err'
--    Right funcs' -> forM_ funcs' print
    Right (func':funcs') -> do
      T.writeFile "ffi/NativeFunctions.hs" [st|
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
        (C.TypeName "at::Tensor", #{bra}t|Tensor|#{cket})
    ]
}

C.include "<ATen/ATen.h>"

cpp_cast_Byte :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
cpp_cast_Byte a b =  #{bra}C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Byte(*$(at::Tensor* a), $(bool b))); }|#{cket}
|]
