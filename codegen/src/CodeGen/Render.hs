{-# LANGUAGE NamedFieldPuns #-}
module CodeGen.Render
  ( makeModule
  , writeHaskellModule

  , parseFile
  , cleanList

  , renderFunctions
  ) where

import Control.Monad (join)
import Control.Arrow ((&&&))
import CodeGen.Prelude
import Data.List
import System.Directory (createDirectoryIfMissing)
import qualified Data.HashMap.Strict as HM
import qualified Data.HashSet as HS
import qualified Data.Text as T

import CodeGen.Types
import CodeGen.Render.Function (renderSig, SigType(..), mkHsname)
import CodeGen.Parse.Cases (checkFunction, type2hsreal)
import qualified CodeGen.Parse as CG (parser)


-- ----------------------------------------
-- helper data and functions for templating
-- ----------------------------------------

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (extensions' <> [""])
 where
  extensions' :: [Text]
  extensions' = renderExtension <$> extensions

  renderExtension :: Text -> Text
  renderExtension extension = "{-# LANGUAGE " <> extension <> " #-}"

renderModule :: HModule -> Text
renderModule m
  = "module "
  <> outModule (lib m)
  <> generatedTypeModule
  <> (case basename of
         "" -> ""
         _  -> "." <> basename)
 where
  basename :: Text
  basename = textFileSuffix (fileSuffix m)

  generatedTypeModule :: Text
  generatedTypeModule = case isTemplate m of
    ConcreteFiles -> ""
    GenericFiles -> "." <> type2hsreal (typeTemplate m)

renderExports :: [Text] -> Text
renderExports _ = " where\n\n"

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" (("import " <>) <$> imports) <> "\n\n"


renderFunctions :: HModule -> [(Maybe (LibType, Text), Function)] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderSig'    IsFun <$> remainder)
    <> (renderSig' IsFunPtr <$> remainder)
 where
  renderSig' t = renderSig t (lib m) (isTemplate m) (header m) (typeTemplate m) (suffix m) (fileSuffix m)

  remainder :: [(Maybe (LibType, Text), Text, Parsable, [Arg])]
  remainder = go <$> validFunctions
    where
      go :: (Maybe (LibType, Text), Function) -> (Maybe (LibType, Text), Text, Parsable, [Arg])
      go (mp, f) = (mp, funName f, funReturn f, funArgs f)


validFunctions :: LibType -> [Function] -> TemplateType -> [(Maybe (LibType, Text), Function)]
validFunctions lt fs tt
  -- ensure that everything is unique (while maintaining the rendered order)
  = nub

  -- use a prefix if the function prefix/namespace doesn't line up with the current module
  $ map ((join . fmap checkLibs . funPrefix) &&& id)

  -- filter any functions which don't belong
  $ filter (checkFunction lt tt . FunctionName . funName) fs

 where
  checkLibs :: (LibType, Text) -> Maybe (LibType, Text)
  checkLibs pref = do
    guard (lt /= fst pref)
    pure pref


withFunctionCounts :: (Function -> x) -> (Maybe (LibType, Text) -> Function -> x) -> (Int, Function) -> x
withFunctionCounts fnEQ fnGT (c, f) = case compare c 1 of
  LT -> impossible "counts must be > 0 since we are counting empirical occurences"
  EQ -> fnEQ f
  GT -> fnGT (funPrefix f) f

renderAll :: HModule -> Text
renderAll m
  =  renderExtensions (extensions m)
  <> renderModule m
  <> renderExports exportFunctions
  <> renderImports (imports m)
  <> renderFunctions m validFunctions'
  where
    validFunctions' :: [(Maybe (LibType, Text), Function)]
    validFunctions' = validFunctions (lib m) (bindings m) (typeTemplate m)

    fun2name :: SigType -> (Maybe (LibType, Text), Function) -> Text
    fun2name st (mp, fn) = mkHsname (lib m) st mp (funName fn)

    exportFunctions :: [Text]
    exportFunctions
      =  fmap (fun2name IsFun)    validFunctions'
      <> fmap (fun2name IsFunPtr) validFunctions'

writeHaskellModule
  :: [Function]
  -> (TemplateType -> [Function] -> HModule)
  -> TemplateType
  -> IO ()
writeHaskellModule parsedBindings makeConfig templateType
  | numFunctions == 0 =
    tputStrLn $ "No bindings found for " <> outDir <> filename

  | otherwise = do
    tputStrLn $ "Writing " <> outDir <> filename
    createDirectoryIfMissing True (T.unpack outDir)
    writeFile (T.unpack $ outDir <> filename) (T.unpack . renderAll $ modSpec)
 where
  modSpec :: HModule
  modSpec = makeConfig templateType parsedBindings

  basename :: Text
  basename = textFileSuffix (fileSuffix modSpec)

  filename :: Text
  filename = case basename of
    "" -> type2hsreal templateType <> ".hs"
    bn -> bn <> ".hs"

  outDir :: Text
  outDir =
    case basename of
     "" -> textPath (modOutDir modSpec) <> "/"
     _  -> textPath (modOutDir modSpec) <> "/" <> type2hsreal templateType <> "/"

  numFunctions :: Int
  numFunctions = length $ validFunctions (lib modSpec) (bindings modSpec) (typeTemplate modSpec)

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- | Remove if list was returned, extract non-Nothing values, o/w empty list
cleanList :: Either (ParseError Char Void) [Maybe Function] -> [Function]
cleanList = either (const []) catMaybes

parseFile :: LibType -> CodeGenType -> String -> IO [Function]
parseFile _ _ file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile CG.parser file
  pure $ cleanList res
 where
  parseFromFile
    :: Parser [Maybe Function]
    -> String
    -> IO (Either (ParseError Char Void) [Maybe Function])
  parseFromFile p file = runParser p file <$> readFile file

