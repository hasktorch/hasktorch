{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (void)
import Data.Monoid ((<>))
import Data.Void
import Data.Text
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Prelude as P
import Text.Show.Pretty


-- ----------------------------------------
-- Parsed types
-- ----------------------------------------

data THType =
  THVoid
  | THDescBuff
  | THTensorPtr
  | THTensorPtrPtr
  | THStoragePtr
  | THLongStoragePtr
  | THPtrDiff
  | THLongPtr
  | THLong
  | THInt
  | THChar
  | THRealPtr
  | THReal
  | THAccRealPtr
  | THAccReal
  deriving Show

data THArg = THArg {
  thArgType :: THType,
  thArgName :: Text
  } deriving Show

data THItem = THSkip
            | THFunction {
                funName :: Text,
                funArgs :: [THArg],
                funReturn :: THType
                } deriving Show

type Parser = Parsec Void String

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
  modBindings :: [THItem]
  } deriving Show

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

thPtr :: Parser Char
thPtr = char '*'

thVoid :: Parser THType
thVoid = string "void" >> pure THVoid

thDescBuff :: Parser THType
thDescBuff = string "THDescBuff" >> pure THDescBuff

thTensorPtr :: Parser THType
thTensorPtr = string "THTensor" >> space >> thPtr >> pure THTensorPtr

thTensorPtrPtr :: Parser THType
-- thTensorPtrPtr = string "THTensor" >> space >> (count 2 thPtr) >> pure THTensorPtrPtr
thTensorPtrPtr = string "THTensor **" >> pure THTensorPtrPtr
-- TODO : clean up pointer matching

thStoragePtr :: Parser THType
thStoragePtr = string "THStorage" >> space >> thPtr >> pure THStoragePtr

thLongStoragePtr :: Parser THType
thLongStoragePtr = string "THLongStorage" >> space >> thPtr >> pure THStoragePtr

thPtrDiff :: Parser THType
thPtrDiff = string "ptrdiff_t" >> pure THStoragePtr

thLongPtr :: Parser THType
thLongPtr = string "long *" >> pure THLongPtr
-- TODO : clean up pointer matching

thLong :: Parser THType
thLong = string "long" >> pure THLong

thInt :: Parser THType
thInt = string "int" >> pure THInt

thChar :: Parser THType
thChar = string "char" >> pure THChar

thRealPtr :: Parser THType
thRealPtr = string "real *" >> pure THRealPtr
-- TODO : clean up pointer matching

thReal :: Parser THType
thReal = string "real" >> pure THReal

thType = do
  ((string "const " >> pure ()) <|> space)
  (thVoid
   <|> thDescBuff
   <|> thTensorPtrPtr -- match ptr ptr before ptr
   <|> thTensorPtr
   <|> thStoragePtr
   <|> thLongStoragePtr
   <|> thPtrDiff
   <|> thLongPtr
   <|> thLong
   <|> thInt
   <|> thChar
   <|> thRealPtr -- ptr before concrete
   <|> thReal)

-- Landmarks

thAPI :: Parser String
thAPI = string "TH_API"

thSemicolon :: Parser Char
thSemicolon = char ';'

-- Function signatures

thFunctionArgVoid = do
  arg <- thVoid
  space
  char ')' :: Parser Char -- TODO move this outside
  pure $ THArg THVoid ""

thFunctionArgNamed = do
  argType <- thType
  space
  argName <- some (alphaNumChar <|> char '_')
  space
  (char ',' :: Parser Char) <|> (char ')' :: Parser Char)
  space
  pure $ THArg argType (T.pack argName)

thFunctionArg = thFunctionArgVoid <|> thFunctionArgNamed

thFunctionArgs = do
  char '(' :: Parser Char
  functionArgs <- some thFunctionArg
  -- close paren consumed by last thFunctionArg (TODO - clean this up)
  pure functionArgs

thFunctionTemplate = do
  thAPI >> space
  funRet <- thType
  space
  string "THTensor_("
  funName <- some alphaNumChar
  space
  string ")"
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ THFunction (T.pack funName) funArgs funRet

thSkip = do
  eol <|> (some (notChar '\n') >> eol)
  pure $ THSkip

thItem = thFunctionTemplate <|> thSkip -- ordering is important

thFile = some thItem

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

renderModule :: Text -> TemplateType -> Text -> Text
renderModule prefix templateType suffix = "module " <> prefix <> (type2SpliceReal templateType) <> suffix

renderExports :: [Text] -> Text
renderExports exports = " (\n    " <> (T.intercalate ",\n    " exports) <> ") where\n\n"

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" $ (\x -> "import " <> x) <$> imports

renderFunName :: Text -> THItem -> Text
renderFunName _ THSkip = ""
renderFunName prefix (THFunction name args ret) =
  prefix <> "_" <> name

renderFunSig :: Text -> THItem -> Text
renderFunSig _ THSkip = ""
renderFunSig prefix (THFunction name args ret) =
  prefix <> "_" <> name <> " :: \n"
  -- TODO signature

renderAll spec =
  -- P.foldr (\x y -> x <> ",\n" <> y) "" (renderFunName . prefix <$> bindings)
  (renderModule (modPrefix spec) (modTypeTemplate spec) (modSuffix spec)
   <> renderExports ((renderFunName $ "c_TH" <> (type2SpliceReal . modTypeTemplate $ spec) <> "Tensor") <$> modBindings spec)
   <> renderImports (modImports spec)
  )
  where
    prefix = makePrefix . type2SpliceReal . modTypeTemplate $ spec
    bindings = modBindings spec

-- ----------------------------------------
-- Execution
-- ----------------------------------------

parseFromFile p file = runParser p file <$> readFile file

cleanList (Left _) = []
cleanList (Right lst) = P.filter f lst
  where
    f THSkip = False
    f _ = True

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
        modImports = ["Foreign", "Foreign.C.Types",
                      "Foreign.C.String", "Foreign.ForeignPtr"],
        modTypeDefs = [],
        modBindings = bindings
  }

main = do
  parsedBindings <- testFile "vendor/torch7/lib/TH/generic/THTensor.h"
  putStrLn "First 5 signatures"
  putStrLn $ ppShow (P.take 5 parsedBindings)
  putStrLn $ ppShow (P.take 5 (renderFunName "THIntTensor" <$> parsedBindings))
  let intModule = makeModule GenInt parsedBindings
  let floatModule = makeModule GenFloat parsedBindings
  putStrLn "Writing THIntTensor.hs"
  writeFile "./render/THIntTensor.hs" (T.unpack . renderAll $ intModule)
  putStrLn "Writing THFloatTensor.hs"
  writeFile "./render/THFloatTensor.hs" (T.unpack . renderAll $ floatModule)
  putStrLn "Done"
