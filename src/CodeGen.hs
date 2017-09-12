module Main where

import Control.Monad (void)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L

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
  funReturn :: THType,
  funName :: String,
  funArgs :: [THArg]
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

sc :: Parser ()
sc = L.space space1 lineCmnt blockCmnt
  where
    lineCmnt  = L.skipLineComment "//"
    blockCmnt = L.skipBlockComment "/*" "*/"

thType = do
  -- (string "const ") <|> sc
  -- (string "const " <|> some (char ' '))
  (thVoid
   <|> thTensorPtr
   <|> thStoragePtr
   <|> thLongStoragePtr)

{- Landmarks -}

thAPI :: Parser String
thAPI = string "TH_API"

thSemicolon :: Parser Char
thSemicolon = char ';'

thFunctionArg = do
  thType
  space
  argName <- some alphaNumChar
  space
  char ',' :: Parser Char
  space

thFunctionArgs = do
  char '(' :: Parser Char
  functionArgs <- some thFunctionArg
  char ')' :: Parser Char
  return functionArgs

thFunctionTemplate = do
  thAPI >> space >> thType >> space
  -- string "THTensor_("
  -- name <- some alphaNumChar
  -- char ')' :: Parser Char
  -- space
  -- thFunctionArgs
  -- thSemicolon

-- testString = "TH_API void THTensor_(clearFlag)(THTensor *self, const char flag);"
testString = "TH_API void"

main = do
  parseTest thFunctionTemplate testString
  putStrLn "Done"
