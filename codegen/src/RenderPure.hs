{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderPure where

import GHC.Generics
import Data.Yaml (ParseException,FromJSON(..))
import qualified Data.Yaml as Y
import Text.Shakespeare.Text (st)
import Data.Text (Text)
import Data.List (isPrefixOf, isSuffixOf, sort)
import Data.Maybe (isJust)
import qualified Data.Text.IO as T
import System.Directory (createDirectoryIfMissing)
import Data.Aeson.Types (defaultOptions, genericParseJSON, constructorTagModifier)

import qualified ParseDeclarations as D
import ParseFunctionSig as P
import RenderCommon

data BindingName
  = HsName String
  | CppName String
--  | RegexHsName String
--  | RegexCppName String
  deriving (Show, Eq, Generic)

data Binding
  = BindRename { src :: BindingName, hs_name :: String }
  | Bind       { src :: BindingName }
  | BindRemove { src :: BindingName }
  deriving (Show, Eq, Generic)

instance FromJSON BindingName where
  parseJSON = genericParseJSON defaultOptions{
    constructorTagModifier = \tag ->
      case tag of
        "HsName" -> "hs_name"
        "CppName" -> "cpp_name"
--        "RegexHsName" -> "regex_hs_name"
--        "RegexCppName" -> "regex_cpp_name"
        a -> a
    }

instance FromJSON Binding where
  parseJSON = genericParseJSON defaultOptions{
    constructorTagModifier = \tag ->
      case tag of
        "BindRename" -> "rename"
        "Bind"       -> "bind"
        "BindRemove" -> "remove"
        a -> a
    }



toFunction :: D.Declaration -> P.Function
toFunction dl = P.Function
  { P.name = D.name dl
  , P.parameters = map (\a -> P.Parameter (D.type2type a) (D.name' a) Nothing) $ D.arguments dl
  , P.retType = case D.returns dl of
      [a] -> D.type2type a
      ax -> P.Tuple $ map D.type2type ax
  , P.variant = P.VFunction
  }

renderFunctions :: [(String, D.Declaration)] -> Text
renderFunctions nfs = mconcat $ flip map nfs $ \(n,nf) -> pureFunction True n (toFunction nf)

isRemove :: Binding ->  Bool
isRemove (BindRemove _) = True
isRemove _ = False

isRename :: Binding ->  Bool
isRename (BindRename _ _) = True
isRename _ = False

removeBinding :: Binding -> (String, D.Declaration) -> Bool
removeBinding (BindRemove (HsName n)) (hsName, _) = n == hsName
removeBinding (BindRemove (CppName n)) (_, d) = n == D.name d
removeBinding _ _ = False

removeBinding' :: [Binding] -> (String, D.Declaration) -> Bool
removeBinding' bindings decl = any (\b -> removeBinding b decl) bindings

removeFilter :: [Binding] -> [(String, D.Declaration)] -> [(String, D.Declaration)]
removeFilter bindings fns = filter (removeBinding' bindings') fns
  where
    bindings' = filter isRemove bindings

renameBinding :: Binding -> (String, D.Declaration) -> Maybe (String, D.Declaration)
renameBinding (BindRename (HsName n) new_name) (hsName, decl) =
  if n == hsName then Just (new_name,decl) else Nothing
renameBinding (BindRename (CppName n) new_name) (_, decl) =
  if n == D.name decl then Just (new_name,decl) else Nothing
renameBinding _ _ = Nothing

renameBinding' :: [Binding] -> (String, D.Declaration) -> (String, D.Declaration)
renameBinding' bindings decl =
  case foldl (\i b -> let v = (renameBinding b decl) in if isJust v then v else i) Nothing bindings of
    Just v -> v
    Nothing -> decl

renameFilter ::  [Binding] -> [(String, D.Declaration)] -> [(String, D.Declaration)]
renameFilter bindings fns = map (renameBinding' bindings') fns
  where
    bindings' = filter isRename bindings

nativeFunctionsFilter :: [D.Declaration] -> [Binding] -> [(String, D.Declaration)]
nativeFunctionsFilter fns bindings =
  filter (\(n,a) ->
            D.mode a == D.Native &&
            "namespace" `elem` (D.method_of a) &&
            D.is_factory_method a == Nothing &&
            not (isPrefixOf "_" (D.name a)) &&
            not (isSuffixOf "_" (D.name a)) &&
            not (isSuffixOf "_out" (D.name a)) &&
            all (/= P.Ptr P.GeneratorType) (map D.dynamic_type' (D.arguments a)) &&
            not ((D.name a) `elem` notUniqList)
--            map D.dynamic_type' (D.returns a) == [P.TenType P.Tensor]
         ) $
  renameFilter bindings $
  removeFilter bindings $
  map (\f -> (getSignatures (toFunction f),f)) fns
  where
    notUniqList :: [String]
    notUniqList = notUniq (sort $ map D.name fns) []
    notUniq [] a = a
    notUniq (x:y:xs) ys = if x == y then notUniq xs (y:ys) else (notUniq (y:xs) ys)
    notUniq a b = b

decodeAndCodeGen :: String -> String -> String -> IO ()
decodeAndCodeGen basedir yamlSpecFileName bindingsFileName = do
  funcs <- Y.decodeFileEither yamlSpecFileName :: IO (Either ParseException [D.Declaration])
  bindings <- Y.decodeFileEither bindingsFileName :: IO (Either ParseException [Binding])
  case (funcs,bindings) of
    (Left err', _) -> print err'
    (Right _  , Left err') -> print err'
    (Right fns, Right bnd) -> do
      createDirectoryIfMissing True (basedir <> "/Torch/Pure")
      T.writeFile (basedir <> "/ATen/Pure/Native.hs") $
        template "ATen.Pure.Native" $
        renderFunctions $ nativeFunctionsFilter fns bnd

renderImport :: Text -> Text
renderImport module_name = [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Managed.Native
import ATen.Managed.Type.Generator
import ATen.Managed.Type.IntArray
import ATen.Managed.Type.Scalar
import ATen.Managed.Type.SparseTensorRef
import ATen.Managed.Type.Storage
import ATen.Managed.Type.Tensor
import ATen.Managed.Type.TensorList
import ATen.Managed.Type.TensorOptions
import ATen.Managed.Type.Tuple
import ATen.Managed.Type.StdString
import ATen.Managed.Type.StdArray
|]

template :: Text -> Text -> Text
template module_name functions = [st|
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module #{module_name} where

#{renderImport module_name}
#{functions}
|]
