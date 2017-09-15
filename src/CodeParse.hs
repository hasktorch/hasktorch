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
  THVoidPtr
  | THVoid
  | THDescBuff
  | THTensorPtrPtr
  | THTensorPtr
  | THByteTensorPtr
  | THLongTensorPtr
  | THDoubleTensorPtr
  | THFloatTensorPtr

  | THGeneratorPtr
  | THStoragePtr
  | THLongStoragePtr
  | THAllocatorPtr
  | THPtrDiff

  | THDouble
  | THLongPtr
  | THLong
  | THIntPtr
  | THInt
  | THSize
  | THCharPtr
  | THChar
  | THRealPtr
  | THReal
  | THAccRealPtr
  | THAccReal
  deriving Show

data THArg = THArg {
  thArgType :: THType,
  thArgName :: Text -- some signatures, e.g. Storage.h have no variable name
  } deriving Show

data THFunction = THFunction {
                funName :: Text,
                funArgs :: [THArg],
                funReturn :: THType
                } deriving Show

type Parser = Parsec Void String

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

thPtr :: Parser Char
thPtr = char '*'

thVoidPtr :: Parser THType
thVoidPtr = (string "void *" <|> string "void*") >> pure THVoidPtr

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

thByteTensorPtr :: Parser THType
thByteTensorPtr = string "THByteTensor" >> space >> thPtr >> pure THByteTensorPtr

thLongTensorPtr :: Parser THType
thLongTensorPtr = string "THLongTensor" >> space >> thPtr >> pure THLongTensorPtr

thDoubleTensorPtr :: Parser THType
thDoubleTensorPtr = string "THDoubleTensor" >> space >> thPtr >> pure THDoubleTensorPtr

thFloatTensorPtr :: Parser THType
thFloatTensorPtr = string "THFloatTensor" >> space >> thPtr >> pure THFloatTensorPtr

thGeneratorPtr :: Parser THType
thGeneratorPtr = string "THGenerator" >> space >> thPtr >> pure THTensorPtr

thStoragePtr :: Parser THType
thStoragePtr = (string "THStorage *" <|> string "THStorage*") >> pure THStoragePtr

thLongStoragePtr :: Parser THType
-- thLongStoragePtr = string "THLongStorage" >> space >> thPtr >> pure THStoragePtr
thLongStoragePtr = (string "THLongStorage *" <|> string "THLongStorage*")
  >> space >> pure THLongStoragePtr

thLongAllocatorPtr :: Parser THType
thLongAllocatorPtr = (string "THAllocator *" <|> string "THAllocator*")
  >> space >> pure THAllocatorPtr

thPtrDiff :: Parser THType
thPtrDiff = string "ptrdiff_t" >> pure THStoragePtr

thLongPtr :: Parser THType
thLongPtr = string "long *" >> pure THLongPtr
-- TODO : clean up pointer matching

thLong :: Parser THType
thLong = string "long" >> pure THLong

thIntPtr :: Parser THType
-- thIntPtr = string "int" >> space >> thPtr >> pure THIntPtr
thIntPtr = (string "int *" <|> string "int* ") >> pure THIntPtr

thInt :: Parser THType
thInt = string "int" >> pure THInt

thSize :: Parser THType
thSize = string "size_t" >> pure THSize

thCharPtr :: Parser THType
thCharPtr = (string "char*" <|> string "char *") >> pure THChar

thChar :: Parser THType
thChar = string "char" >> pure THChar

thRealPtr :: Parser THType
thRealPtr = (string "real *" <|> string "real* ") >> pure THRealPtr
-- TODO : clean up pointer matching

thReal :: Parser THType
thReal = string "real" >> pure THReal

thAccReal :: Parser THType
thAccReal = string "accreal" >> pure THAccReal

thAccRealPtr :: Parser THType
thAccRealPtr = string "accreal *" >> pure THAccRealPtr

thType = do
  ((string "const " >> pure ()) <|> space)
  (
    thVoidPtr
    <|> thVoid
    <|> thDescBuff
    <|> thTensorPtrPtr -- match ptr ptr before ptr
    <|> thTensorPtr
    <|> thByteTensorPtr
    <|> thLongTensorPtr
    <|> thDoubleTensorPtr
    <|> thFloatTensorPtr
    <|> thGeneratorPtr
    <|> thStoragePtr
    <|> thLongStoragePtr
    <|> thLongAllocatorPtr
    <|> thDouble
    <|> thPtrDiff
    <|> thLongPtr
    <|> thLong
    <|> thIntPtr
    <|> thInt
    <|> thSize
    <|> thCharPtr
    <|> thChar
    <|> thRealPtr
    <|> thReal
    <|> thAccRealPtr
    <|> thAccReal
    )

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
  argName <- (some (alphaNumChar <|> char '_')) <|> string "" -- e.g. Storage.h - no variable name
  space
  (char ',' :: Parser Char) <|> (char ')' :: Parser Char)
  space
  pure $ THArg argType (T.pack argName)

thFunctionArg = thFunctionArgNamed <|> thFunctionArgVoid

thFunctionArgs = do
  char '(' :: Parser Char
  functionArgs <- some thFunctionArg
  -- close paren consumed by last thFunctionArg (TODO - clean this up)
  pure functionArgs

thFunctionPrefixes = string "THTensor_("
                     <|> string "THBlas_("
                     <|> string "THLapack_("
                     <|> string "THStorage_("
                     <|> string "THVector_("

thFunctionTemplate = do
  thAPI >> space
  funRet <- thType
  space
  thFunctionPrefixes
  funName <- some (alphaNumChar <|> char '_')
  space
  string ")"
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ Just $ THFunction (T.pack funName) funArgs funRet

-- TODO - exclude thAPI so it doesn't parse if TH_API is invalid
thSkip = do
  eol <|> (some (notChar '\n') >> eol)
  pure Nothing

thItem = thFunctionTemplate <|> thSkip -- ordering is important

thFile = some thItem
