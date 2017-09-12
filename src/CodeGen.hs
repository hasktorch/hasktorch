{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (void)
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
  thArgName :: String
  } deriving Show

data THItem = THSkip
            | THFunction {
                funName :: String,
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
  modTypeTemplate :: TemplateType,
  modExtensions :: [Text],
  modImports :: [Text],
  modTypeDefs :: [(Text, Text)],
  modBindings :: [THItem]
  } deriving Show

-- ----------------------------------------
-- Templated header parser
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

-- Landmark parsing

thAPI :: Parser String
thAPI = string "TH_API"

thSemicolon :: Parser Char
thSemicolon = char ';'

-- Function parsing

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
  pure $ THArg argType argName

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
  pure $ THFunction funName funArgs funRet

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

makePrefix templateType = "TH" ++ templateType ++ "Tensor"

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

renderFunName _ THSkip = ""
renderFunName prefix (THFunction name args ret) =
  prefix ++ "_" ++ name

renderFunSig _ THSkip = ""
renderFunSig prefix (THFunction name args ret) =
  prefix ++ "_" ++ name ++ " :: \n"
  -- TODO signature

renderAll templateType lst =
  P.foldr (\x y -> x ++ ",\n" ++ y) "" (renderFunName prefix <$> lst)
  where prefix = makePrefix templateType

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
  -- case res of
  --   Left err -> putStr (parseErrorPretty err)
  --   Right val -> putStr $ (ppShow val) ++ "\n"

test1 = do
  testString ex1
  where
    ex1 = "skip this garbage line line\n" ++
     "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" ++
     "another garbage line ( )@#R @# 324 32"

main = do
  res <- testFile "vendor/torch7/lib/TH/generic/THTensor.h"
  putStrLn "First 5 signatures"
  putStrLn $ ppShow (P.take 5 res)
  putStrLn $ ppShow (P.take 5 (renderFunName "THIntTensor" <$> res))
  putStrLn "Writing test.hs"
  writeFile "./render/test.hs" (renderAll "Int" res)
  putStrLn "Done"
