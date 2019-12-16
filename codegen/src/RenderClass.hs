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
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified #{"Torch.Internal.Unmanaged.Type." <> (PC.hsnameWithoutSpace typ_)} as Unmanaged
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
import Torch.Internal.Type
import Torch.Internal.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"
|]


renderConstructors :: Bool -> PC.CppClassSpec -> Text
renderConstructors is_managed typ_ = mconcat $ map (methodToCpp typ_ True is_managed True "" "") (PC.constructors typ_)

renderDestructor :: Bool -> PC.CppClassSpec -> Text
renderDestructor is_managed typ_ = if is_managed then "" else [st|
delete#{PC.hsnameWithoutSpace typ_} :: Ptr #{PC.hsnameWithParens typ_} -> IO ()
delete#{PC.hsnameWithoutSpace typ_} object = #{bra}C.throwBlock| void { delete $(#{PC.cppname typ_}* object);}|#{cket}

instance CppObject #{PC.hsnameWithParens typ_} where
  fromPtr ptr = newForeignPtr ptr (delete#{PC.hsnameWithoutSpace typ_} ptr)
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
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Unmanaged/Type")
      T.writeFile (basedir <> "/Torch/Internal/Unmanaged/Type/" <> PC.hsnameWithoutSpace fns <> ".hs") $
        template False ("Torch.Internal.Unmanaged.Type." <> fromString (PC.hsnameWithoutSpace fns)) fns
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Managed/Type")
      T.writeFile (basedir <> "/Torch/Internal/Managed/Type/" <> PC.hsnameWithoutSpace fns <> ".hs") $
        template True ("Torch.Internal.Managed.Type." <> fromString (PC.hsnameWithoutSpace fns)) fns


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

