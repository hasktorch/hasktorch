{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RenderClass where

import           Control.Monad         (forM_)
import           Data.String           (fromString)
import           Data.Text             (Text, unpack)
import qualified Data.Text.IO          as T
import           Data.Yaml             (ParseException)
import qualified Data.Yaml             as Y
import           System.Directory      (createDirectoryIfMissing)
import           Text.Shakespeare.Text (st)

import qualified ParseClass   as PC
import           RenderCommon

renderImport :: Bool -> PC.CppClassSpec -> Text -> Text
renderImport is_managed typ_ unmanagedModuleName =  if is_managed then  [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified #{unmanagedModuleName} as Unmanaged
|] else [st|
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

|]


renderConstructors :: Bool -> PC.CppClassSpec -> [Text]
renderConstructors is_managed typ_ = map (methodToCpp typ_ True is_managed True "" "") (PC.constructors typ_)

renderHeaders :: Bool -> PC.CppClassSpec -> Text
renderHeaders True _ = ""
renderHeaders False typ_ = mconcat $ map (\s -> [st|C.include "<#{s}>"
|]) (PC.headers typ_)

renderMethods :: Bool -> PC.CppClassSpec -> [Text]
renderMethods is_managed typ_ = map (methodToCpp typ_ False is_managed True "" "") (PC.methods typ_)

renderFunctions :: Bool -> PC.CppClassSpec -> [Text]
renderFunctions is_managed typ_ = map (functionToCpp is_managed True "at::" "") (PC.functions typ_)

decodeAndCodeGen :: String -> String -> Int -> IO ()
decodeAndCodeGen basedir fileName 1 = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException PC.CppClassSpec)
  case funcs of
    Left err' -> print err'
    Right spec -> do
      let moduleName = PC.hsnameWithoutSpace spec
          fns is_managed =
            renderConstructors is_managed spec ++
            renderMethods is_managed spec ++
            renderFunctions is_managed spec
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Unmanaged/Type")
      T.writeFile (basedir <> "/Torch/Internal/Unmanaged/Type/" <> moduleName <> ".hs") $
        template
          False
          ("Torch.Internal.Unmanaged.Type." <> fromString (moduleName))
          ""
          spec
          (fns False)
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Managed/Type")
      T.writeFile (basedir <> "/Torch/Internal/Managed/Type/" <> moduleName <> ".hs") $
        template
          True
          ("Torch.Internal.Managed.Type." <> fromString (moduleName))
          ("Torch.Internal.Unmanaged.Type." <> fromString (moduleName))
          spec
          (fns True)
decodeAndCodeGen basedir fileName num = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException PC.CppClassSpec)
  case funcs of
    Left err' -> print err'
    Right spec -> do
      let moduleName = PC.hsnameWithoutSpace spec
          fns is_managed =
            renderConstructors is_managed spec ++
            renderMethods is_managed spec ++
            renderFunctions is_managed spec
          unmanagedFuncs = split' num (fns False)
          managedFuncs = split' num (fns True)
      createDirectoryIfMissing True (unpack [st|#{basedir}/Torch/Internal/Unmanaged/Type/#{moduleName}|])
      forM_ (zip [0..] unmanagedFuncs) $ \(i::Int,fns') -> do
        T.writeFile (unpack [st|#{basedir}/Torch/Internal/Unmanaged/Type/#{moduleName}/#{moduleName}#{i}.hs|]) $
          template
            False
            [st|Torch.Internal.Unmanaged.Type.#{moduleName}.#{moduleName}#{i}|]
            ""
            spec
            fns'
        createDirectoryIfMissing True (unpack [st|#{basedir}/Torch/Internal/Managed/Type/#{moduleName}|])
      forM_ (zip [0..] managedFuncs) $ \(i::Int,fns') -> do
        T.writeFile (unpack [st|#{basedir}/Torch/Internal/Managed/Type/#{moduleName}/#{moduleName}#{i}.hs|]) $
          template
            True
            [st|Torch.Internal.Managed.Type.#{moduleName}.#{moduleName}#{i}|]
            [st|Torch.Internal.Unmanaged.Type.#{moduleName}.#{moduleName}#{i}|]
            spec
            fns'


template :: Bool -> Text -> Text -> PC.CppClassSpec -> [Text] -> Text
template is_managed module_name unamagedModuleName types funcs = [st|
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module #{module_name} where

#{renderImport is_managed types unamagedModuleName}

#{renderHeaders is_managed types}

#{mconcat funcs}
|]
