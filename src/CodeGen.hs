{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (void)
import Data.Monoid ((<>))
import Data.Maybe
import Data.Void
import Data.Text
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Prelude as P
import Text.Show.Pretty

import CodeParse

-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data TemplateType = GenByte
                  | GenChar
                  | GenDouble
                  | GenFloat
                  | GenHalf
                  | GenInt
                  | GenLong
                  | GenShort deriving Show

data HModule = HModule {
  modPrefix :: Text,
  modTypeTemplate :: TemplateType,
  modSuffix :: Text,
  modExtensions :: [Text],
  modImports :: [Text],
  modTypeDefs :: [(Text, Text)],
  modBindings :: [THFunction]
  } deriving Show

testString inp = case (parse thFile "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

-- ----------------------------------------
-- Rendering
-- ----------------------------------------

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal GenByte   = "Byte"
type2SpliceReal GenChar   = "Byte"
type2SpliceReal GenDouble = "Double"
type2SpliceReal GenFloat  = "Float"
type2SpliceReal GenHalf   = "Half"
type2SpliceReal GenInt    = "Int"
type2SpliceReal GenLong   = "Long"
type2SpliceReal GenShort  = "Short"

-- #define real [X]
type2real :: TemplateType -> Text
type2real GenByte   = "unsigned char"
type2real GenChar   = "char"
type2real GenDouble = "double"
type2real GenFloat  = "float"
type2real GenHalf   = "THHalf"
type2real GenInt    = "int"
type2real GenLong   = "long"
type2real GenShort  = "short"

-- #define accreal [X]
type2accreal :: TemplateType -> Text
type2accreal GenByte   = "long"
type2accreal GenChar   = "long"
type2accreal GenDouble = "double"
type2accreal GenFloat  = "double"
type2accreal GenHalf   = "float"
type2accreal GenInt    = "long"
type2accreal GenLong   = "long"
type2accreal GenShort  = "long"

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> "#-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (renderExtension <$> extensions)

renderModuleName :: Text -> Text -> TemplateType -> Text
renderModuleName prefix suffix templateType =
  prefix <> (type2SpliceReal templateType) <> suffix

renderModule :: Text -> Text -> TemplateType -> Text
renderModule prefix suffix templateType =
  "module " <> (renderModuleName prefix suffix templateType) 

renderExports :: [Text] -> Text
renderExports exports = (" (\n    "
                         <> (T.intercalate ",\n    " exports)
                         <> ") where\n\n")

renderImports :: [Text] -> Text
renderImports imports = (T.intercalate "\n" (singleimport <$> imports)) <> "\n\n"
  where singleimport x = "import " <> x

renderFunName :: Text -> Text -> Text
renderFunName prefix name = prefix <> "_" <> name

renderFunSig :: Text -> Text -> Text
renderFunSig prefix name  =
  ("foreign import ccall \"THTensor.h" <> name <> "\"\n"
   <> prefix <> "_" <> name <> " :: \n")
  -- TODO signature

renderFunctions :: [THFunction] -> Text
renderFunctions bindings =
  -- TODO fix up function names
  intercalate "\n\n" ((renderFunSig "c_TH") <$> (funName <$> bindings))

renderAll :: HModule -> Text
renderAll spec =
  (
    (renderModule (modPrefix spec) (modSuffix spec) (modTypeTemplate spec))
    <> renderExports (
    (renderFunName ("c_TH" <> (type2SpliceReal . modTypeTemplate $ spec) <> "Tensor"))
    <$> (fmap funName (modBindings spec)))
    <> renderImports (modImports spec)
    <> renderFunctions (modBindings spec)
  )
  where
    prefix = makePrefix . type2SpliceReal . modTypeTemplate $ spec
    bindings = modBindings spec

-- ----------------------------------------
-- Execution
-- ----------------------------------------

parseFromFile p file = runParser p file <$> readFile file

cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList (Left _) = []
cleanList (Right lst) = fromJust <$> (P.filter f lst)
  where
    f Nothing = False
    f (Just _) = True

testFile file = do
  res <- parseFromFile thFile file
  pure $ cleanList res

test1 = do
  testString ex1
  where
    ex1 = "skip this garbage line line\n" <>
     "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
     "another garbage line ( )@#R @# 324 32"

makeModule typeTemplate bindings =
   HModule {
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = "Tensor",
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types"],
        modTypeDefs = [],
        modBindings = bindings
  }

renderTensorFile templateType parsedBindings = do
  let filename = (renderModuleName "TH" "Tensor" templateType) <> ".hs"
  let modspec = makeModule templateType parsedBindings
  putStrLn $ "Writing " <> T.unpack filename
  writeFile ("./render/" ++ T.unpack filename) (T.unpack . renderAll $ modspec)

genTypes = [GenByte, GenChar,
            GenDouble, GenFloat, GenHalf,
            GenInt, GenLong, GenShort] :: [TemplateType]

runTensor = do
  parsedBindings <- testFile "vendor/torch7/lib/TH/generic/THTensor.h"
  putStrLn "First 3 signatures"
  putStrLn $ ppShow (P.take 3 parsedBindings)
  mapM_ (\x -> renderTensorFile x parsedBindings) genTypes

main = do
  runTensor
  putStrLn "Done"
