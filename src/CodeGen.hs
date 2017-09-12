module Main where

import Control.Monad (void)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Text.Show.Pretty

data THType =
  THVoid
  | THTensorPtr
  | THStoragePtr
  | THLongStoragePtr
  | THPtrDiff
  | THLong
  | THInt
  | THChar
  | THReal deriving Show

data THArg = THArg {
  thArgType :: THType,
  thArgName :: String
  } deriving Show

data THFunction = THFunction {
  funName :: String,
  funArgs :: [THArg],
  funReturn :: THType
  } deriving Show

type Parser = Parsec Void String

{- Type Parsers -}

thPtr :: Parser Char
thPtr = char '*'

thVoid :: Parser THType
thVoid = do
  string "void"
  return THVoid

thTensorPtr = do
  string "THTensor"
  space
  thPtr
  return THTensorPtr

thStoragePtr = do
  string "THStorage"
  space
  thPtr
  return THStoragePtr

thLongStoragePtr = do
  string "THLongStorage" >> space >> thPtr
  return THStoragePtr

thLong :: Parser THType
thLong = do
  string "long"
  return THLong

thInt :: Parser THType
thInt = do
  string "int"
  return THInt

thChar :: Parser THType
thChar = do
  string "char"
  return THChar

thReal :: Parser THType
thReal = do
  string "real"
  return THChar

sc :: Parser ()
sc = L.space space1 lineCmnt blockCmnt
  where
    lineCmnt  = L.skipLineComment "//"
    blockCmnt = L.skipBlockComment "/*" "*/"

thType = do
  ((string "const " >> pure ()) <|> space)
  -- (string "const " <|> some (char ' '))
  (thVoid
   <|> thTensorPtr
   <|> thStoragePtr
   <|> thLongStoragePtr
   <|> thLong
   <|> thInt
   <|> thChar
   <|> thReal)

{- Landmarks -}

thAPI :: Parser String
thAPI = string "TH_API"

thSemicolon :: Parser Char
thSemicolon = char ';'

thFunctionArg = do
  argType <- thType
  space
  argName <- some alphaNumChar
  space
  (char ',' :: Parser Char) <|> (char ')' :: Parser Char)
  space
  pure $ THArg argType argName

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

test inp = case (parse thFunctionTemplate "" inp) of
  Left err -> putStr (parseErrorPretty err)
  Right val -> putStr $ (ppShow val) ++ "\n"

main = do
  test testString1
  putStrLn "Done"
  where
    testString1 = "TH_API void THTensor_(setFlag)(THTensor *self, const char flag);"
