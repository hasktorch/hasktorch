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
import CodeGen.Parse (thParser)
import CodeGen.Parse.Cases (checkFunction, signatureAliases, type2hsreal, type2real)
import qualified CodeGen.Render.Haskell as Hs


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
renderModule moduleSpec = "module " <> renderModuleName moduleSpec

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
renderFunctions :: HModule -> [THFunction] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderSig' IsFun    <$> triple)
    <> (renderSig' IsFunPtr <$> triple)
 where
  renderSig' t = renderSig t (isTemplate m) (header m) (typeTemplate m) (suffix m)

  triple :: [(Text, THType, [THArg])]
  triple = go <$> validFunctions
    where
      go :: THFunction -> (Text, THType, [THArg])
      go f = (funName f, funReturn f, funArgs f)

-- | Check for conditional templating of functions and filter function list
checkList :: [THFunction] -> TemplateType -> [THFunction]
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
    validFunctions :: [THFunction]
    validFunctions = checkList (bindings m) (typeTemplate m)

    fun2name :: Text -> THFunction -> Text
    fun2name p = (\f -> p <> "_" <> f) . funName

    exportFunctions :: [Text]
    exportFunctions
      =  fmap (fun2name "c") validFunctions
      <> fmap (fun2name "p") validFunctions

writeHaskellModule
  :: [THFunction]
  -> (TemplateType -> [THFunction] -> HModule)
  -> TemplateType
  -> IO ()
writeHaskellModule parsedBindings makeConfig templateType = do
  tputStrLn $ "Writing " <> outDir <> filename
  createDirectoryIfMissing True (T.unpack outDir)
  writeFile (T.unpack $ outDir <> filename) (T.unpack . renderAll $ modSpec)
 where
  modSpec :: HModule
  modSpec = makeConfig templateType parsedBindings

  filename :: Text
  filename = renderModuleName modSpec <> ".hs"

  outDir :: Text
  outDir = textPath (modOutDir modSpec) <> "/" <> type2hsreal templateType <> "/"

renderModuleName :: HModule -> Text
renderModuleName HModule{prefix, typeTemplate, fileSuffix}
  = T.pack (show prefix) <> type2real typeTemplate <> textFileSuffix fileSuffix

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- | Remove if list was returned, extract non-Nothing values, o/w empty list
cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList = either (const []) catMaybes

parseFile :: CodeGenType -> String -> IO [THFunction]
parseFile cgt file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile (thParser cgt) file
  pure $ cleanList res
 where
  parseFromFile
    :: Parser [Maybe THFunction]
    -> String
    -> IO (Either (ParseError Char Void) [Maybe THFunction])
  parseFromFile p file = runParser p file <$> readFile file

