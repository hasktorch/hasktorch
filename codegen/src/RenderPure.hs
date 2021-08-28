{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RenderPure where

import Data.Aeson.Types -- (defaultOptions, genericParseJSON, constructorTagModifier, sumEncoding(..))
import Data.List (isPrefixOf, isSuffixOf, sort)
import Data.Maybe (isJust)
import Data.Set (fromList, member)
import Data.Text (Text)
import qualified Data.Text.IO as T
import qualified Data.Yaml as Y
import GHC.Generics
import qualified ParseDeclarations as D
import ParseFunctionSig as P
import RenderCommon
import System.Directory (createDirectoryIfMissing)
import Text.Shakespeare.Text (st)

data Binding
  = BindRename {src :: String, dst :: String}
  | Bind {src :: String}
  | BindRemove {src :: String}
  deriving (Show, Eq, Generic)

instance FromJSON Binding where
  parseJSON =
    genericParseJSON
      defaultOptions
        { sumEncoding = ObjectWithSingleField,
          allNullaryToStringTag = True,
          constructorTagModifier = \tag ->
            case tag of
              "BindRename" -> "rename"
              "Bind" -> "bind"
              "BindRemove" -> "remove"
              a -> a
        }

toFunction :: D.Declaration -> P.Function
toFunction dl =
  P.Function
    { P.name = D.name dl,
      P.parameters = map (\a -> P.Parameter (D.type2type a) (D.name' a) Nothing) $ D.arguments dl,
      P.retType = case D.returns dl of
        [a] -> D.type2type a
        ax -> P.Tuple $ map D.type2type ax,
      P.variant = P.VFunction
    }

renderFunctions :: [(String, D.Declaration)] -> Text
renderFunctions nfs = mconcat $ flip map nfs $ \(n, nf) -> pureFunction n (toFunction nf)

isRemove :: Binding -> Bool
isRemove (BindRemove _) = True
isRemove _ = False

isRename :: Binding -> Bool
isRename (BindRename _ _) = True
isRename _ = False

removeBinding :: Binding -> (String, D.Declaration) -> Bool
removeBinding (BindRemove n) (hsName, _) = n == hsName
removeBinding _ _ = False

removeBinding' :: [Binding] -> (String, D.Declaration) -> Bool
removeBinding' bindings decl = any (\b -> removeBinding b decl) bindings

removeFilter :: [Binding] -> [(String, D.Declaration)] -> [(String, D.Declaration)]
removeFilter bindings fns = filter (\v -> not (removeBinding' bindings' v)) fns
  where
    bindings' = filter isRemove bindings

renameBinding :: Binding -> (String, D.Declaration) -> Maybe (String, D.Declaration)
renameBinding (BindRename n new_name) (hsName, decl) =
  if n == hsName then Just (new_name, decl) else Nothing
renameBinding _ _ = Nothing

renameBinding' :: [Binding] -> (String, D.Declaration) -> (String, D.Declaration)
renameBinding' bindings decl =
  case foldl (\i b -> let v = (renameBinding b decl) in if isJust v then v else i) Nothing bindings of
    Just v -> v
    Nothing -> decl

renameFilter :: [Binding] -> [(String, D.Declaration)] -> [(String, D.Declaration)]
renameFilter bindings fns = unmask <$> renamed
  where
    bindings' = filter isRename bindings
    renamed = map (renameBinding' bindings') fns
    dupeSet = fromList . notUniqList $ D.name . snd <$> renamed
    unmask = \(k, d) -> let k' = D.name d in if member k' dupeSet then (k, d) else (k', d)

nativeFunctionsFilter :: [D.Declaration] -> [Binding] -> [(String, D.Declaration)]
nativeFunctionsFilter fns bindings =
  filter
    ( \(_, a) ->
        D.mode a == D.Native
          && "namespace" `elem` (D.method_of a)
          && D.is_factory_method a == Just False
          && not (isPrefixOf "_" (D.name a))
          && not (isSuffixOf "_" (D.name a))
          && not (isSuffixOf "_out" (D.name a))
          && not (isSuffixOf "_backward" (D.name a))
          && all (/= P.GeneratorType) (map D.dynamic_type' (D.arguments a))
    )
    $ renameFilter bindings $
      removeFilter bindings $
        map (\f -> (getSignatures (toFunction f), f)) fns

notUniqList :: [String] -> [String]
notUniqList lst = notUniq (sort lst) []
  where
    notUniq [] a = a
    notUniq (x : y : xs) ys = if x == y then notUniq xs (y : ys) else (notUniq (y : xs) ys)
    notUniq _ b = b

decodeAndCodeGen :: String -> String -> String -> IO ()
decodeAndCodeGen basedir yamlSpecFileName bindingsFileName = do
  funcs <- Y.decodeFileEither yamlSpecFileName :: IO (Either Y.ParseException [D.Declaration])
  bindings <- Y.decodeFileEither bindingsFileName :: IO (Either Y.ParseException [Binding])
  case (funcs, bindings) of
    (Left err', _) -> print err'
    (Right _, Left err') -> print err'
    (Right fns, Right bnd) -> do
      createDirectoryIfMissing True (basedir <> "/Torch/Functional/")
      let l = nativeFunctionsFilter fns bnd
      T.writeFile (basedir <> "/Torch/Functional/Internal.hs") $
        template "Torch.Functional.Internal" $
          renderFunctions l

renderImport :: Text
renderImport =
  [st|
import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Managed.Type.Tuple as ATen
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.Cast
import Torch.Internal.Cast

import Torch.Tensor
import Torch.Scalar
import Torch.Dimname
import Torch.DType
import Torch.Cast
|]

template :: Text -> Text -> Text
template module_name functions =
  [st|
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module #{module_name} where

#{renderImport}
#{functions}
|]
