{-# LANGUAGE OverloadedStrings #-}

module CodeParse (
  Parser,
  thFile,
  THType(..),
  THArg(..),
  THFunction(..)
  ) where

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

-- ----------------------------------------
-- Parsed types
-- ----------------------------------------

data THType =
  THVoid
  | THDescBuff
  | THTensorPtr
  | THTensorPtrPtr
  | THGeneratorPtr
  | THStoragePtr
  | THLongStoragePtr
  | THPtrDiff
  | THDouble
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

data THFunction = THFunction {
                funName :: Text,
                funArgs :: [THArg],
                funReturn :: THType
                } deriving Show

-- data THItem = THItemSkip | THItemFunction THFunction deriving Show

type Parser = Parsec Void String

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

thPtr :: Parser Char
thPtr = char '*'

thVoid :: Parser THType
thVoid = string "void" >> pure THVoid

thDouble :: Parser THType
thDouble = string "double" >> pure THDouble

thDescBuff :: Parser THType
thDescBuff = string "THDescBuff" >> pure THDescBuff

thTensorPtr :: Parser THType
thTensorPtr = string "THTensor" >> space >> thPtr >> pure THTensorPtr

thTensorPtrPtr :: Parser THType
-- thTensorPtrPtr = string "THTensor" >> space >> (count 2 thPtr) >> pure THTensorPtrPtr
thTensorPtrPtr = string "THTensor **" >> pure THTensorPtrPtr
-- TODO : clean up pointer matching

thGeneratorPtr :: Parser THType
thGeneratorPtr = string "THGenerator" >> space >> thPtr >> pure THTensorPtr

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
   <|> thGeneratorPtr
   <|> thStoragePtr
   <|> thLongStoragePtr
   <|> thDouble
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
  funName <- some (alphaNumChar <|> char '_')
  space
  string ")"
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ Just $ THFunction (T.pack funName) funArgs funRet

thSkip = do
  eol <|> (some (notChar '\n') >> eol)
  pure Nothing

thItem = thFunctionTemplate <|> thSkip -- ordering is important

thFile = some thItem

