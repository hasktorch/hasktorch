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
import System.Directory (createDirectoryIfMissing)

import qualified ParseClass as PC
import RenderCommon

renderImport :: Bool -> PC.CppClassSpec -> Text
renderImport is_managed typ_ =  if is_managed then  [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.SparseTensorRef
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple

import qualified #{"ATen.Unmanaged.Type." <> (PC.hsname typ_)} as Unmanaged
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
import ATen.Type
import ATen.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"
|]

renderConstructors :: Bool -> PC.CppClassSpec -> Text
renderConstructors is_managed typ_ = mconcat $ map (methodToCpp typ_ True is_managed True "" "") (PC.constructors typ_)

renderDestructor :: Bool -> PC.CppClassSpec -> Text
renderDestructor is_managed typ_ = if is_managed then "" else [st|
delete#{PC.hsname typ_} :: Ptr #{PC.hsname typ_} -> IO ()
delete#{PC.hsname typ_} object = #{bra}C.throwBlock| void { delete $(#{PC.cppname typ_}* object);}|#{cket}

instance CppObject #{PC.hsname typ_} where
  fromPtr ptr = newForeignPtr ptr (delete#{PC.hsname typ_} ptr)
|]


renderMethods :: Bool -> PC.CppClassSpec -> Text
renderMethods is_managed typ_ = mconcat $ map (methodToCpp typ_ False is_managed True "" "") (PC.methods typ_)

renderFunctions :: Bool -> PC.CppClassSpec -> Text
renderFunctions is_managed typ_ = mconcat $ map (functionToCpp is_managed True "at::" "") (PC.functions typ_)

decodeAndCodeGen :: String -> String -> IO ()
decodeAndCodeGen basedir fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException PC.CppClassSpec)
  case funcs of
    Left err' -> print err'
    Right fns -> do
      print fns
      createDirectoryIfMissing True (basedir <> "/ATen/Unmanaged/Type")
      T.writeFile (basedir <> "/ATen/Unmanaged/Type/" <> PC.hsname fns <> ".hs") $
        template False ("ATen.Unmanaged.Type." <> fromString (PC.hsname fns)) fns
      createDirectoryIfMissing True (basedir <> "/ATen/Managed/Type")
      T.writeFile (basedir <> "/ATen/Managed/Type/" <> PC.hsname fns <> ".hs") $
        template True ("ATen.Managed.Type." <> fromString (PC.hsname fns)) fns


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

#{renderFunctions is_managed types}
|]

