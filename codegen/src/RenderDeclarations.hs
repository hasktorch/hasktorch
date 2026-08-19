{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RenderDeclarations where

import Control.Monad (forM_)
import qualified Data.Set as S
import Data.Text (Text, replace, unpack)
import qualified Data.Text.IO as T
import Data.Yaml (ParseException)
import qualified Data.Yaml as Y
import qualified ParseDeclarations as D
import ParseFunctionSig as P
import RenderCommon
import System.Directory (createDirectoryIfMissing)
import Text.Shakespeare.Text (st)

dropGenerator :: [Parameter] -> [Parameter]
dropGenerator params = filter (\v' -> ptype v' /= Ptr GeneratorType) params

addFunctionWithDefaultArguments :: D.Declaration -> [D.Declaration]
addFunctionWithDefaultArguments dl = map (\args -> dl {D.arguments = args}) genArgsWithDefault'
  where
    genArgsWithDefault [] = []
    genArgsWithDefault (x : xs)
      | D.default' x /= Nothing = (x : xs) : genArgsWithDefault xs
      | otherwise = [(x : xs)]
    genArgsWithDefault' =
      case D.arguments dl of
        [] -> [[]]
        xs -> map reverse $ genArgsWithDefault (reverse xs)

toFunction :: D.Declaration -> P.Function
toFunction dl =
  P.Function
    { P.name = D.name dl,
      P.parameters = map (\a -> P.Parameter (D.type2type a) (D.name' a) Nothing) $ D.arguments dl,
      P.retType = case D.returns dl of
        [a] -> D.type2type a
        [] -> P.CType P.CVoid
        ax -> P.Tuple $ map D.type2type ax,
      P.variant = P.VFunction
    }

uniqFilter :: Ord n => (a -> n) -> [a] -> [a]
uniqFilter item xs = uniqFilter' xs S.empty
  where
    uniqFilter' [] _ = []
    uniqFilter' (x : xs') ids = if S.member (item x) ids then uniqFilter' xs' ids else x : (uniqFilter' xs (S.insert (item x) ids))

renderFunctions :: Bool -> Bool -> String -> [D.Declaration] -> Text
renderFunctions is_managed enb_type_initials namespace nfs =
  mconcat $
    map (\nf -> functionToCpp is_managed enb_type_initials namespace "" nf) $
      uniqFilter getSignatures $
        map toFunction nfs

decodeAndCodeGen :: String -> String -> IO ()
decodeAndCodeGen basedir fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException [D.Declaration])
  case funcs of
    Left err' -> print err'
    Right fns' -> do
      generateNativeFunctions fns'
  where
    generateNativeFunctions fns' = do
      let fns = concat $ map addFunctionWithDefaultArguments fns'
      createDirectoryIfMissing True (basedir <> "/Torch")
      createDirectoryIfMissing True (basedir <> "/Torch/Internal")

      let nativeFunctions = filter (\a -> D.mode a == D.Native && "namespace" `elem` (D.method_of a)) fns
          nativeFunctions' = split' 16 nativeFunctions

      forM_ (zip [0 ..] nativeFunctions') $ \(i :: Int, funcs') -> do
        createDirectoryIfMissing True (basedir <> "/Torch/Internal/Unmanaged/Native")
        createDirectoryIfMissing True (basedir <> "/Torch/Internal/Managed/Native")
        T.writeFile (unpack [st|#{basedir}/Torch/Internal/Managed/Native/Native#{i}.hs|]) $
          template False True [st|Torch.Internal.Managed.Native.Native#{i}|] $
            renderFunctions True True "at::" funcs'
        T.writeFile (unpack [st|#{basedir}/Torch/Internal/Unmanaged/Native/Native#{i}.hs|]) $
          template False False [st|Torch.Internal.Unmanaged.Native.Native#{i}|] $
            renderFunctions False True "at::" funcs'

      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Managed")
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Unmanaged")
      T.writeFile (basedir <> "/Torch/Internal/Managed/TensorFactories.hs") $
        template True True "Torch.Internal.Managed.TensorFactories" $
          renderFunctions True True "torch::" (filter (\a -> D.mode a == D.Native && "namespace" `elem` (D.method_of a) && D.is_factory_method a == Just True) fns)
      T.writeFile (basedir <> "/Torch/Internal/Unmanaged/TensorFactories.hs") $
        template True False "Torch.Internal.Unmanaged.TensorFactories" $
          renderFunctions False True "torch::" (filter (\a -> D.mode a == D.Native && "namespace" `elem` (D.method_of a) && D.is_factory_method a == Just True) fns)

renderImport :: Bool -> Bool -> Text -> Text
renderImport is_torch_factory_method is_managed module_name =
  if is_managed
    then
      [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified #{replace "Managed" "Unmanaged" module_name} as Unmanaged
|]
    else
      [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"
|]
        <> if is_torch_factory_method
          then
            [st|
C.include "<torch/csrc/autograd/generated/variable_factories.h>"
|]
          else ""

template :: Bool -> Bool -> Text -> Text -> Text
template is_torch_factory_method is_managed module_name functions =
  [st|
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module #{module_name} where

#{renderImport is_torch_factory_method is_managed module_name}
#{functions}
|]
