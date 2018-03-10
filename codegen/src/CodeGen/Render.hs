{-# LANGUAGE NamedFieldPuns #-}
module CodeGen.Render
  ( makeModule
  , writeHaskellModule

  , parseFile
  , cleanList
  ) where

import CodeGen.Prelude
import qualified Data.Text as T
import System.Directory (createDirectoryIfMissing)

import CodeGen.Types
import CodeGen.Render.Function (renderSig, SigType(..))
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
  <> outModule (prefix m)
  <> generatedTypeModule
  <> "." <> textFileSuffix (fileSuffix m)
 where
  generatedTypeModule :: Text
  generatedTypeModule = case isTemplate m of
    ConcreteFiles -> ""
    GenericFiles -> "." <> type2hsreal (typeTemplate m)

renderExports :: [Text] -> Text
renderExports exports = T.intercalate "\n"
  [ ""
  , "  ( " <> T.intercalate "\n  , " exports
  , "  ) where"
  , ""
  , ""
  ]

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" (("import " <>) <$> imports) <> "\n\n"


-- TODO clean up redundancy of valid functions vs. functions in moduleSpec
renderFunctions :: HModule -> [Function] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderSig' IsFun    <$> triple)
    <> (renderSig' IsFunPtr <$> triple)
 where
  renderSig' t = renderSig t (prefix m) (isTemplate m) (header m) (typeTemplate m) (suffix m)

  triple :: [(Text, Parsable, [Arg])]
  triple = go <$> validFunctions
    where
      go :: Function -> (Text, Parsable, [Arg])
      go f = (funName f, funReturn f, funArgs f)

-- | Check for conditional templating of functions and filter function list
checkList :: [Function] -> TemplateType -> [Function]
checkList fList templateType =
  filter (checkFunction templateType . FunctionName . funName) fList

renderAll :: HModule -> Text
renderAll m
  =  renderExtensions (extensions m)
  <> renderModule m
  <> renderExports exportFunctions
  <> renderImports (imports m)
  <> renderFunctions m validFunctions
  where
    validFunctions :: [Function]
    validFunctions = checkList (bindings m) (typeTemplate m)

    fun2name :: Text -> Function -> Text
    fun2name p = (\f -> p <> "_" <> f) . funName

    exportFunctions :: [Text]
    exportFunctions
      =  fmap (fun2name "c") validFunctions
      <> fmap (fun2name "p") validFunctions

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

  filename :: Text
  filename = textFileSuffix (fileSuffix modSpec) <> ".hs"

  outDir :: Text
  outDir = textPath (modOutDir modSpec) <> "/" <> type2hsreal templateType <> "/"

  numFunctions :: Int
  numFunctions = length $ checkList (bindings modSpec) (typeTemplate modSpec)

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

