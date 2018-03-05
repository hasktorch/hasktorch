{-# LANGUAGE NamedFieldPuns #-}
module CodeGen.Render
  ( makeModule
  , renderCHeaderFile

  , parseFile
  , cleanList
  ) where

import CodeGen.Prelude
import qualified Data.Text as T

import CodeGen.Types
import CodeGen.Render.Function (renderFunPtrSig, renderFunSig)
import CodeGenParse (thParser)
import ConditionalCases (checkFunction, signatureAliases)
import qualified CodeGen.Render.Haskell as Hs

makeModule
  :: LibType
  -> TextPath
  -> CodeGenType
  -> FilePath
  -> ModuleSuffix
  -> FileSuffix
  -> TemplateType
  -> [THFunction]
  -> HModule
makeModule a00 a0 a1 a2 a3 a4 a5 a6
  = HModule
  { prefix = a00
  , extensions = ["ForeignFunctionInterface"]
  , imports = ["Foreign", "Foreign.C.Types", "THTypes", "Data.Word", "Data.Int"]
  , typeDefs = []
  , modOutDir = a0
  , isTemplate = a1
  , header = a2
  , suffix = a3
  , fileSuffix = a4
  , typeTemplate = a5
  , bindings = a6
  }

-- makeTHModule :: TextPath -> IsTemplate -> FilePath -> ModuleSuffix -> FileSuffix -> TemplateType -> [THFunction] -> HModule
-- makeTHModule = makeModule TH

-- ----------------------------------------
-- helper data and functions for templating
-- ----------------------------------------

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> " #-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (extensions' <> [""])
 where
  extensions' :: [Text]
  extensions' = renderExtension <$> extensions

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
    $  (renderFunSig'    <$> triple)
    <> (renderFunPtrSig' <$> triple)
 where
  renderFunSig'    = renderFunSig    (isTemplate m) ffiPrefix (header m) (typeTemplate m)
  renderFunPtrSig' = renderFunPtrSig (isTemplate m) ffiPrefix (header m) (typeTemplate m)

  ffiPrefix :: Text
  ffiPrefix
    = T.pack (show $ prefix m)
    <> Hs.type2SpliceReal (typeTemplate m)
    <> textSuffix (suffix m)

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

renderCHeaderFile
  :: [THFunction] -> (TemplateType -> [THFunction] -> HModule) -> TemplateType -> IO ()
renderCHeaderFile parsedBindings makeConfig templateType = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile (outDir ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
 where
  modSpec :: HModule
  modSpec = makeConfig templateType parsedBindings

  filename :: Text
  filename = renderModuleName modSpec <> ".hs"

  outDir :: String
  outDir = T.unpack (textPath $ modOutDir modSpec)

renderModuleName :: HModule -> Text
renderModuleName HModule{prefix, typeTemplate, fileSuffix}
  = T.pack (show prefix) <> Hs.type2SpliceReal typeTemplate <> textFileSuffix fileSuffix

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- |Remove If list was returned, extract non-Nothing values, o/w empty list
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

