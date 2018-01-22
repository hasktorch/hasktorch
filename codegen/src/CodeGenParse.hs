{-# LANGUAGE OverloadedStrings #-}

module CodeGenParse (
  Parser,
  thParseGeneric,
  thParseConcrete,
  THType(..),
  THArg(..),
  THFunction(..)
  ) where

import Data.Functor.Identity

import Data.Maybe
import Data.Void
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Prelude as P

import CodeGenTypes

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

thPtr :: Parser Char
thPtr = char '*'

thVoidPtr :: Parser THType
thVoidPtr = (string "void *" <|> string "void*") >> pure THVoidPtr

thBool :: Parser THType
thBool = string "bool" >> pure THBool

thVoid :: Parser THType
thVoid = string "void" >> pure THVoid

thFloatPtr :: Parser THType
thFloatPtr = (string "float *" <|> string "float* ") >> pure THFloatPtr

thFloat :: Parser THType
thFloat = string "float" >> pure THFloat

thDoublePtr :: Parser THType
thDoublePtr = (string "double *" <|> string "double* ") >> pure THDoublePtr

thDouble :: Parser THType
thDouble = string "double" >> pure THDouble

thDescBuff :: Parser THType
thDescBuff = string "THDescBuff" >> pure THDescBuff

{- NN types -}

thNNStatePtr :: Parser THType
thNNStatePtr = string "THNNState" >> space >> thPtr >> pure THNNStatePtr

thIndexTensorPtr :: Parser THType
thIndexTensorPtr = string "THIndexTensor" >> space >> thPtr >> pure THIndexTensorPtr

thIntegerTensorPtr :: Parser THType
thIntegerTensorPtr = string "THIntegerTensor" >> space >> thPtr >> pure THIntegerTensorPtr

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
-- thGeneratorPtr = string "THGenerator" >> space >> thPtr >> pure THGeneratorPtr
thGeneratorPtr = (string "THGenerator * "  <|> string "THGenerator *" <|> string "THGenerator* ") >> pure THGeneratorPtr

thLongAllocatorPtr :: Parser THType
thLongAllocatorPtr = (string "THAllocator *" <|> string "THAllocator*")
  >> space >> pure THAllocatorPtr

thPtrDiff :: Parser THType
thPtrDiff = string "ptrdiff_t" >> pure THPtrDiff

thLongPtrPtr :: Parser THType
thLongPtrPtr = string "long **" >> pure THLongPtrPtr

thLongPtr :: Parser THType
thLongPtr = string "long *" <|> string "long* " >> pure THLongPtr
-- TODO : clean up pointer matching

thLong :: Parser THType
thLong = string "long" >> pure THLong

thIntPtr :: Parser THType
-- thIntPtr = string "int" >> space >> thPtr >> pure THIntPtr
thIntPtr = (string "int *" <|> string "int* ") >> pure THIntPtr

thInt :: Parser THType
thInt = string "int" >> pure THInt

thUInt64 :: Parser THType
thUInt64 = string "uint64_t" >> pure THUInt64

thUInt64Ptr :: Parser THType
thUInt64Ptr = (string "uint64_t *" <|> string "uint64_t* ") >> pure THUInt64Ptr

thUInt64PtrPtr :: Parser THType
thUInt64PtrPtr = (string "uint64_t **" <|> string "uint64_t** ") >> pure THUInt64PtrPtr

thUInt32 :: Parser THType
thUInt32 = string "uint32_t" >> pure THInt32

thUInt32Ptr :: Parser THType
thUInt32Ptr = (string "uint32_t *" <|> string "uint32_t* ") >> pure THInt32Ptr

thUInt32PtrPtr :: Parser THType
thUInt32PtrPtr = (string "uint32_t **" <|> string "uint32_t** ") >> pure THInt32PtrPtr

thUInt16 :: Parser THType
thUInt16 = string "uint16_t" >> pure THInt16

thUInt16Ptr :: Parser THType
thUInt16Ptr = (string "uint16_t *" <|> string "uint16_t* ") >> pure THInt16Ptr

thUInt16PtrPtr :: Parser THType
thUInt16PtrPtr = (string "uint16_t **" <|> string "uint16_t** ") >> pure THInt16PtrPtr

thUInt8 :: Parser THType
thUInt8 = string "uint8_t" >> pure THInt8

thUInt8Ptr :: Parser THType
thUInt8Ptr = (string "uint8_t *" <|> string "uint8_t* ") >> pure THInt8Ptr

thUInt8PtrPtr :: Parser THType
thUInt8PtrPtr = (string "uint8_t **" <|> string "uint8_t** ") >> pure THInt8PtrPtr



thInt64 :: Parser THType
thInt64 = string "int64_t" >> pure THInt64

thInt64Ptr :: Parser THType
thInt64Ptr = (string "int64_t *" <|> string "int64_t* ") >> pure THInt64Ptr

thInt64PtrPtr :: Parser THType
thInt64PtrPtr = (string "int64_t **" <|> string "int64_t** ") >> pure THInt64PtrPtr

thInt32 :: Parser THType
thInt32 = string "int32_t" >> pure THInt32

thInt32Ptr :: Parser THType
thInt32Ptr = (string "int32_t *" <|> string "int32_t* ") >> pure THInt32Ptr

thInt32PtrPtr :: Parser THType
thInt32PtrPtr = (string "int32_t **" <|> string "int32_t** ") >> pure THInt32PtrPtr

thInt16 :: Parser THType
thInt16 = string "int16_t" >> pure THInt16

thInt16Ptr :: Parser THType
thInt16Ptr = (string "int16_t *" <|> string "int16_t* ") >> pure THInt16Ptr

thInt16PtrPtr :: Parser THType
thInt16PtrPtr = (string "int16_t **" <|> string "int16_t** ") >> pure THInt16PtrPtr

thInt8 :: Parser THType
thInt8 = string "int8_t" >> pure THInt8

thInt8Ptr :: Parser THType
thInt8Ptr = (string "int8_t *" <|> string "int8_t* ") >> pure THInt8Ptr

thInt8PtrPtr :: Parser THType
thInt8PtrPtr = (string "int8_t **" <|> string "int8_t** ") >> pure THInt8PtrPtr





thSize :: Parser THType
thSize = string "size_t" >> pure THSize

thCharPtrPtr :: Parser THType
thCharPtrPtr = (string "char**" <|> string "char **") >> pure THCharPtrPtr

thCharPtr :: Parser THType
thCharPtr = (string "char*" <|> string "char *") >> pure THCharPtr

thChar :: Parser THType
thChar = string "char" >> pure THChar

thShortPtr :: Parser THType
thShortPtr = (string "short *" <|> string "short* ") >> pure (THShortPtr)

thShort :: Parser THType
thShort = string "short" >> pure THShort

thHalfPtr :: Parser THType
thHalfPtr = (string "THHalf *" <|> string "THHalf* ") >> pure (THHalfPtr)

thHalf :: Parser THType
thHalf = string "THHalf" >> pure THHalf

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

-- not meant to be a complete C spec, just enough for TH lib
thType :: ParsecT Void String Identity THType
thType = do
  ((string "const " >> pure ())
    <|> (string "unsigned " >> pure ())
    <|> (string "struct " >> pure ()) -- See THStorageCopy.h
    <|> space)
  (
    -- pointers take precedence in parsing
    thVoidPtr
    <|> thVoid
    <|> thDescBuff

    <|> thBool

    <|> thNNStatePtr
    <|> thIndexTensorPtr
    <|> thIntegerTensorPtr

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
    <|> thFloatPtr
    <|> thFloat
    <|> thDoublePtr
    <|> thDouble
    <|> thPtrDiff
    <|> thLongPtrPtr
    <|> thLongPtr
    <|> thLong

    <|> thUInt64PtrPtr
    <|> thUInt64Ptr
    <|> thUInt64
    <|> thUInt32PtrPtr
    <|> thUInt32Ptr
    <|> thUInt32
    <|> thUInt16PtrPtr
    <|> thUInt16Ptr
    <|> thUInt16
    <|> thUInt8PtrPtr
    <|> thUInt8Ptr
    <|> thUInt8

    <|> thInt64PtrPtr -- must come before thInt*
    <|> thInt64Ptr
    <|> thInt64
    <|> thInt32PtrPtr
    <|> thInt32Ptr
    <|> thInt32
    <|> thInt16PtrPtr
    <|> thInt16Ptr
    <|> thInt16
    <|> thInt8PtrPtr
    <|> thInt8Ptr
    <|> thInt8
    <|> thIntPtr
    <|> thInt
    <|> thSize
    <|> thCharPtrPtr
    <|> thCharPtr
    <|> thChar
    <|> thShortPtr
    <|> thShort
    <|> thHalfPtr
    <|> thHalf
    <|> thRealPtr
    <|> thReal
    <|> thAccRealPtr
    <|> thAccReal

    <|> thFilePtr
    )

-- Landmarks

thAPI :: Parser String
thAPI = string "TH_API"
-- thAPI = string "TH_AP" >> char 'o'
-- thAPI = char 'T' >> char 'H' >> char 'A' >> char '_' >> char 'P' >> char 'I'

thSemicolon :: Parser Char
thSemicolon = char ';'

-- Function signatures

thFunctionArgVoid :: Parser THArg
thFunctionArgVoid = do
  arg <- thVoid
  space
  char ')' :: Parser Char -- TODO move this outside
  pure $ THArg THVoid ""

thFunctionArgNamed :: Parser THArg
thFunctionArgNamed = do
  argType <- thType
  --space <|> (space >> string "volatile" >> space)
  space
  -- e.g. declaration sometimes has no variable name - eg Storage.h
  argName <- (some (alphaNumChar <|> char '_')) <|> string ""
  space
  (char ',' :: Parser Char) <|> (char ')' :: Parser Char)
  space
  pure $ THArg argType (T.pack argName)

thFunctionArg :: Parser THArg
thFunctionArg = thFunctionArgNamed <|> thFunctionArgVoid

thFunctionArgs :: Parser [THArg]
thFunctionArgs = do
  char '(' :: Parser Char
  functionArgs <- some thFunctionArg
  -- close paren consumed by last thFunctionArg (TODO - clean this up)
  pure functionArgs

thGenericPrefixes :: Parser String
thGenericPrefixes = string "THTensor_("
                     <|> string "THBlas_("
                     <|> string "THLapack_("
                     <|> string "THStorage_("
                     <|> string "THVector_("
                     <|> string "THNN_("

thFunctionTemplate :: Parser (Maybe THFunction)
thFunctionTemplate = do
  thAPI >> space
  funReturn' <- thType
  space
  thGenericPrefixes
  funName' <- some (alphaNumChar <|> char '_')
  space
  string ")"
  space
  funArgs' <- thFunctionArgs
  thSemicolon
  optional $ try thComment
  pure $ Just $ THFunction (T.pack funName') funArgs' funReturn'

thInlineComment :: Parser ()
thInlineComment = do
  some space
  string "//"
  some (alphaNumChar <|> char '_' <|> char ' ')
  eol <|> (some (notChar '\n') >> eol)
  pure ()

thComment :: Parser ()
--  :: ParsecT Void String Data.Functor.Identity.Identity (Maybe a)
thComment = do
  space
  string "/*"
  some (alphaNumChar <|> char '_' <|> char ' ')
  string "*/"
  pure ()

thFunctionConcrete :: Parser (Maybe THFunction)
thFunctionConcrete = do
  funReturn' <- thType
  space
  funName' <- some (alphaNumChar <|> char '_')
  space
  funArgs' <- thFunctionArgs
  thSemicolon
  optional $ try thComment
  pure $ Just $ THFunction (T.pack funName') funArgs' funReturn'

-- notTHAPI = do
--   x <- manyTill anyChar (try whitespace)

-- TODO - exclude TH_API prefix. Parse should crash if TH_API parse is invalid
thSkip :: Parser (Maybe THFunction)
thSkip = do
  -- x <- manyTill anyChar (try whitespace)
  -- if x == "TH_API"
  eol <|> (some (notChar '\n') >> eol)
  -- eol <|> ((not <?> (string "TH_API")) >> eol)
  pure Nothing

thConstant :: Parser (Maybe THFunction)
thConstant = do
  -- THLogAdd has constants, these are not surfaced
  thAPI >> space
  string "const" >> space
  thType >> space
  (some (alphaNumChar <|> char '_')) >> char ';'
  pure Nothing

thItem :: Parser (Maybe THFunction)
thItem = try thConstant <|> thFunctionTemplate <|> thSkip -- ordering is important

thParseGeneric :: Parser [Maybe THFunction]
thParseGeneric = some thItem

thParseConcrete :: Parser [Maybe THFunction]
thParseConcrete = some (try thConstant <|> (thAPI >> space >> thFunctionConcrete) <|> thSkip)
