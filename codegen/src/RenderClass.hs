{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderClass where

import Data.Yaml (ParseException)
import qualified Data.Yaml as Y
import Text.Shakespeare.Text (st)
import Data.Text (Text)
import Data.String (fromString)
import qualified Data.Text.IO as T
import qualified Data.Text as T
import System.Directory (createDirectoryIfMissing)

import qualified ParseClass as PC
import ParseFunctionSig as P
import RenderCommon

renderImport :: Bool -> PC.CppClassSpec -> Text
renderImport is_managed typ =  if is_managed then  [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import Aten.Unmanaged.Type.Generator
import Aten.Unmanaged.Type.IntArray
import Aten.Unmanaged.Type.Scalar
import Aten.Unmanaged.Type.SparseTensorRef
import Aten.Unmanaged.Type.Storage
import Aten.Unmanaged.Type.Tensor
import Aten.Unmanaged.Type.TensorList
import Aten.Unmanaged.Type.TensorOptions
import Aten.Unmanaged.Type.Tuple

import qualified #{"Aten.Unmanaged.Type." <> (PC.hsname typ)} as Unmanaged
|] else [st|
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
|]

renderConstructors :: Bool -> PC.CppClassSpec -> Text
renderConstructors is_managed typ = mconcat $ map (methodToCpp typ True is_managed False "" "") (PC.constructors typ)

renderDestructor :: Bool -> PC.CppClassSpec -> Text
renderDestructor is_managed typ = if is_managed then "" else [st|
delete#{PC.hsname typ} :: Ptr #{PC.hsname typ} -> IO ()
delete#{PC.hsname typ} object = #{bra}C.block| void { delete $(#{PC.cppname typ}* object);}|#{cket}

instance CppObject #{PC.hsname typ} where
  fromPtr ptr = newForeignPtr ptr (delete#{PC.hsname typ} ptr)
|]


renderMethods :: Bool -> PC.CppClassSpec -> Text
renderMethods is_managed typ = mconcat $ map (methodToCpp typ False is_managed False "" "") (PC.methods typ)


decodeAndCodeGen :: String -> String -> IO ()
decodeAndCodeGen basedir fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException PC.CppClassSpec)
  case funcs of
    Left err' -> print err'
    Right fns -> do
      print fns
      createDirectoryIfMissing True (basedir <> "/Aten/Unmanaged/Type")
      T.writeFile (basedir <> "/Aten/Unmanaged/Type/" <> PC.hsname fns <> ".hs") $
        template False ("Aten.Unmanaged.Type." <> fromString (PC.hsname fns)) fns
      createDirectoryIfMissing True (basedir <> "/Aten/Managed/Type")
      T.writeFile (basedir <> "/Aten/Managed/Type/" <> PC.hsname fns <> ".hs") $
        template True ("Aten.Managed.Type." <> fromString (PC.hsname fns)) fns


template :: Bool -> Text -> PC.CppClassSpec -> Text
template is_managed module_name types = [st|
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module #{module_name} where

#{renderImport is_managed types}

#{renderConstructors is_managed types}

#{renderDestructor is_managed types}

#{renderMethods is_managed types}
|]

