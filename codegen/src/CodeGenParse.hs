{-# LANGUAGE OverloadedStrings #-}

module CodeGenParse (
  Parser,
  thParseGeneric,
  thParseConcrete,
  THType(..),
  THArg(..),
  THFunction(..)
  ) where

import Control.Monad (void)
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

import CodeGenTypes

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

{- Tensor types -}

thTensorPtr :: Parser THType
thTensorPtr = string "THTensor" >> space >> thPtr >> pure THTensorPtr

thTensorPtrPtr :: Parser THType
-- thTensorPtrPtr = string "THTensor" >> space >> (count 2 thPtr) >> pure THTensorPtrPtr
thTensorPtrPtr = string "THTensor **" >> pure THTensorPtrPtr
-- TODO : clean up pointer matching

thByteTensorPtr :: Parser THType
thByteTensorPtr = string "THByteTensor" >> space >> thPtr >> pure THByteTensorPtr

thShortTensorPtr :: Parser THType
thShortTensorPtr = string "THShortTensor" >> space >> thPtr >> pure THShortTensorPtr

thIntTensorPtr :: Parser THType
thIntTensorPtr = string "THIntTensor" >> space >> thPtr >> pure THIntTensorPtr

thLongTensorPtr :: Parser THType
thLongTensorPtr = string "THLongTensor" >> space >> thPtr >> pure THLongTensorPtr

thHalfTensorPtr :: Parser THType
thHalfTensorPtr = string "THHalfTensor" >> space >> thPtr >> pure THHalfTensorPtr

thCharTensorPtr :: Parser THType
thCharTensorPtr = string "THCharTensor" >> space >> thPtr >> pure THCharTensorPtr

thDoubleTensorPtr :: Parser THType
thDoubleTensorPtr = string "THDoubleTensor" >> space >> thPtr >> pure THDoubleTensorPtr

thFloatTensorPtr :: Parser THType
thFloatTensorPtr = string "THFloatTensor" >> space >> thPtr >> pure THFloatTensorPtr

{- Storage -}

thStoragePtr :: Parser THType
thStoragePtr = (string "THStorage *" <|> string "THStorage*") >> pure THStoragePtr

thByteStoragePtr :: Parser THType
thByteStoragePtr = (string "THByteStorage *" <|> string "THByteStorage*")
  >> space >> pure THByteStoragePtr

thCharStoragePtr :: Parser THType
thCharStoragePtr = (string "THCharStorage *" <|> string "THCharStorage*")
  >> space >> pure THCharStoragePtr

thShortStoragePtr :: Parser THType
thShortStoragePtr = (string "THShortStorage *" <|> string "THShortStorage*")
  >> space >> pure THShortStoragePtr

thIntStoragePtr :: Parser THType
thIntStoragePtr = (string "THIntStorage *" <|> string "THIntStorage*")
  >> space >> pure THIntStoragePtr

thHalfStoragePtr :: Parser THType
thHalfStoragePtr = (string "THHalfStorage *" <|> string "THHalfStorage*")
  >> space >> pure THHalfStoragePtr


thLongStoragePtr :: Parser THType
-- thLongStoragePtr = string "THLongStorage" >> space >> thPtr >> pure THStoragePtr
thLongStoragePtr = (string "THLongStorage *" <|> string "THLongStorage*")
  >> space >> pure THLongStoragePtr

thFloatStoragePtr :: Parser THType
thFloatStoragePtr = (string "THFloatStorage *" <|> string "THFloatStorage*")
  >> space >> pure THFloatStoragePtr

thDoubleStoragePtr :: Parser THType
thDoubleStoragePtr = (string "THDoubleStorage *" <|> string "THDoubleStorage*")
  >> space >> pure THDoubleStoragePtr

{- Other -}

thGeneratorPtr :: Parser THType
thGeneratorPtr = string "THGenerator" >> space >> thPtr >> pure THGeneratorPtr

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

thFilePtr :: Parser THType
thFilePtr = (string "THFile *" <|> string "THFile*") >> pure THFilePtr

thType = do
  ((string "const " >> pure ())
    <|> (string "struct " >> pure ()) -- See THStorageCopy.h
    <|> space)
  (
    -- pointers take precedence in parsing
    thVoidPtr
    <|> thVoid
    <|> thDescBuff

    <|> thTensorPtrPtr
    <|> thTensorPtr

    <|> thByteTensorPtr
    <|> thShortTensorPtr
    <|> thIntTensorPtr
    <|> thLongTensorPtr
    <|> thHalfTensorPtr
    <|> thCharTensorPtr
    <|> thDoubleTensorPtr
    <|> thFloatTensorPtr

    <|> thGeneratorPtr

    <|> thStoragePtr
    <|> thByteStoragePtr
    <|> thCharStoragePtr
    <|> thShortStoragePtr
    <|> thIntStoragePtr
    <|> thLongStoragePtr
    <|> thHalfStoragePtr
    <|> thFloatStoragePtr
    <|> thDoubleStoragePtr

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

    <|> thFilePtr
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
  -- e.g. declaration sometimes has no variable name - eg Storage.h
  argName <- (some (alphaNumChar <|> char '_')) <|> string ""
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

thGenericPrefixes = string "THTensor_("
                     <|> string "THBlas_("
                     <|> string "THLapack_("
                     <|> string "THStorage_("
                     <|> string "THVector_("

thFunctionTemplate = do
  thAPI >> space
  funRet <- thType
  space
  thGenericPrefixes
  funName <- some (alphaNumChar <|> char '_')
  space
  string ")"
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ Just $ THFunction (T.pack funName) funArgs funRet

thFunctionConcrete = do
  thAPI >> space
  funRet <- thType
  space
  funName <- some (alphaNumChar <|> char '_')
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ Just $ THFunction (T.pack funName) funArgs funRet

-- TODO - exclude TH_API prefix. Parse should crash if TH_API parse is invalid
thSkip = do
  eol <|> (some (notChar '\n') >> eol)
  pure Nothing

thItem = thFunctionTemplate <|> thSkip -- ordering is important

thParseGeneric = some thItem

thParseConcrete = some (thFunctionConcrete <|> thSkip)
