{-# LANGUAGE TupleSections #-}
{-# LANGUAGE NamedFieldPuns #-}
module CodeGen.Render
  ( makeModule
  , writeHaskellModule

  , parseFile
  , cleanList

  , renderFunctions
  ) where

import CodeGen.Prelude
import Data.List
import System.Directory (createDirectoryIfMissing)
import qualified Data.HashMap.Strict as HM
import qualified Data.HashSet as HS
import qualified Data.Text as T

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
  <> outModule (lib m)
  <> generatedTypeModule
  <> "." <> textFileSuffix (fileSuffix m)
 where
  generatedTypeModule :: Text
  generatedTypeModule = case isTemplate m of
    ConcreteFiles -> ""
    GenericFiles -> "." <> type2hsreal (typeTemplate m)

renderExports :: [Text] -> Text
renderExports _ = " where\n\n"

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" (("import " <>) <$> imports) <> "\n\n"


renderFunctions :: HModule -> [(Int, Function)] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderSig'    IsFun <$> triple)
    <> (renderSig' IsFunPtr <$> triple)
 where
  renderSig' t = renderSig t (lib m) (isTemplate m) (header m) (typeTemplate m) (suffix m) (fileSuffix m)

  triple :: [(Text, Parsable, [Arg])]
  triple = go <$> validFunctions
    where
      go :: (Int, Function) -> (Text, Parsable, [Arg])
      go = withFunctionCounts
        (\f    -> (funName f, funReturn f, funArgs f))
        (\mp f -> (fromMaybe "" mp <> funName f, funReturn f, funArgs f))


validFunctions :: [Function] -> TemplateType -> [(Int, Function)]
validFunctions fs tt
  -- return the number of duplicates function names alongside the function
  = concatMap (\hs -> let es = HS.toList hs; l = length es in (l,) <$> es) $ HM.elems
  -- fold functions into a hashmap grouped by function name
  $ foldr (\f -> HM.insertWith HS.union (funName f) (HS.singleton f)) mempty
  -- filter any functions which don't belong
  $ filter (checkFunction tt . FunctionName . funName) fs

withFunctionCounts :: (Function -> x) -> (Maybe Text -> Function -> x) -> (Int, Function) -> x
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
    validFunctions' :: [(Int, Function)]
    validFunctions' = validFunctions (bindings m) (typeTemplate m)

    fun2name :: Text -> (Int, Function) -> Text
    fun2name p = withFunctionCounts
      (\f    -> p <> "_" <> funName f)
      (\mp f -> p <> "_" <> fromMaybe "" mp <> funName f)

    exportFunctions :: [Text]
    exportFunctions
      =  fmap (fun2name "c") validFunctions'
      <> fmap (fun2name "p") validFunctions'

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
  numFunctions = sum $ map fst $ validFunctions (bindings modSpec) (typeTemplate modSpec)

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

