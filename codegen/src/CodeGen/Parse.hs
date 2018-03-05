module CodeGen.Parse
  ( Parser
  , thParser
  , THType(..)
  , THArg(..)
  , THFunction(..)
  ) where

import Data.Functor.Identity

import Data.Maybe
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import Data.Monoid ((<>))
import Prelude as P
import Data.Text as T

import CodeGen.Types

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

thPtr :: String -> THType -> Parser THType
thPtr s t = do
  string (s <> " * ") <|> string (s <> " *") <|> string (s <> "*") <|> string (s <> "* ")
  pure t

thCPtr :: Parser Char
thCPtr = char '*'

thVoidPtr, thBool, thVoid :: Parser THType
thVoidPtr = thPtr "void" (APtr THVoid)
thBool    = string "bool" >> pure THBool
thVoid    = string "void" >> pure THVoid

thFloatPtr, thFloat, thDoublePtr, thDouble, thDescBuff :: Parser THType
thFloatPtr  = thPtr "float" (APtr THFloat)
thFloat     = string "float" >> pure THFloat
thDoublePtr = thPtr "double" (APtr THDouble)
thDouble    = string "double" >> pure THDouble
thDescBuff  = string "THDescBuff" >> pure THDescBuff

{- NN types -}

-- thNNStatePtr :: Parser THType
-- thNNStatePtr = string "THNNState" >> space >> thCPtr >> pure (APtr THNNState)

-- thIndexTensorPtr :: Parser THType
-- thIndexTensorPtr = string "THIndexTensor" >> space >> thCPtr >> pure (APtr THIndexTensor)

-- thIntegerTensorPtr :: Parser THType
-- thIntegerTensorPtr = string "THIntegerTensor" >> space >> thCPtr >> pure (APtr THIntegerTensor)

{- Tensor types -}

thTensorPtr :: Parser THType
thTensorPtr = string "THTensor" >> space >> thCPtr >> pure (APtr THTensor)

thTensorPtrPtr :: Parser THType
-- thTensorPtrPtr = string "THTensor" >> space >> (count 2 thPtr) >> pure THTensorPtrPtr
thTensorPtrPtr = string "THTensor **" >> pure (APtr (APtr THTensor))
-- TODO : clean up pointer matching

thByteTensorPtr, thIntTensorPtr,    thLongTensorPtr, thHalfTensorPtr :: Parser THType
thCharTensorPtr, thDoubleTensorPtr, thFloatTensorPtr                 :: Parser THType
thByteTensorPtr   = thPtr "THByteTensor" (APtr THByteTensor)
thShortTensorPtr  = thPtr "THShortTensor" (APtr THShortTensor)
thIntTensorPtr    = thPtr "THIntTensor" (APtr THIntTensor)
thLongTensorPtr   = thPtr "THLongTensor" (APtr THLongTensor)
thHalfTensorPtr   = thPtr "THHalfTensor" (APtr THHalfTensor)
thCharTensorPtr   = thPtr "THCharTensor" (APtr THCharTensor)
thDoubleTensorPtr = thPtr "THDoubleTensor" (APtr THDoubleTensor)
thFloatTensorPtr  = thPtr "THFloatTensor" (APtr THFloatTensor)

{- Storage -}

thStoragePtr,    thByteStoragePtr, thCharStoragePtr, thShortStoragePtr :: Parser THType
thIntStoragePtr, thHalfStoragePtr, thLongStoragePtr, thFloatStoragePtr :: Parser THType
thDoubleStoragePtr :: Parser THType

thStoragePtr       = thPtr "THStorage" (APtr THStorage)
thByteStoragePtr   = thPtr "THByteStorage" (APtr THByteStorage)
thCharStoragePtr   = thPtr "THCharStorage" (APtr THCharStorage)
thShortStoragePtr  = thPtr "THShortStorage" (APtr THShortStorage)
thIntStoragePtr    = thPtr "THIntStorage" (APtr THIntStorage)
thHalfStoragePtr   = thPtr "THHalfStorage" (APtr THHalfStorage)
thLongStoragePtr   = thPtr "THLongStorage" (APtr THLongStorage)
thFloatStoragePtr  = thPtr "THFloatStorage" (APtr THFloatStorage)
thDoubleStoragePtr = thPtr "THDoubleStorage" (APtr THDoubleStorage)

{- Other -}

thGeneratorPtr :: Parser THType
-- thGeneratorPtr = string "THGenerator" >> space >> thCPtr  >> pure THGeneratorPtr
thGeneratorPtr = (string "THGenerator * "  <|> string "THGenerator *" <|> string "THGenerator* ") >> pure (APtr THGenerator)

thLongAllocatorPtr :: Parser THType
thLongAllocatorPtr = thPtr "THAllocator" (APtr THAllocator)

thPtrDiff :: Parser THType
thPtrDiff = string "ptrdiff_t" >> pure THPtrDiff

thLongPtrPtr :: Parser THType
thLongPtrPtr = string "long **" >> pure (APtr (APtr THLong))

thLongPtr :: Parser THType
thLongPtr = thPtr "long" (APtr THLong)
-- TODO : clean up pointer matching

thLong, thIntPtr, thInt, thUInt64, thUInt64Ptr, thUInt64PtrPtr, thUInt32 :: Parser THType
thLong         = string "long" >> pure THLong
thIntPtr       = thPtr "int" (APtr THInt)
thInt          = string "int" >> pure THInt
thUInt64       = string "uint64_t" >> pure THUInt64
thUInt64Ptr    = (string "uint64_t *" <|> string "uint64_t* ") >> pure (APtr THUInt64)
thUInt64PtrPtr = (string "uint64_t **" <|> string "uint64_t** ") >> pure (APtr (APtr THUInt64))
thUInt32       = string "uint32_t" >> pure THInt32

thUInt32Ptr, thUInt32PtrPtr, thUInt16, thUInt16Ptr, thUInt16PtrPtr :: Parser THType
thUInt32Ptr    = (string "uint32_t *"  <|> string "uint32_t* ")  >> pure (APtr THInt32)
thUInt32PtrPtr = (string "uint32_t **" <|> string "uint32_t** ") >> pure (APtr (APtr THInt32))
thUInt16       =  string "uint16_t"                              >> pure THInt16
thUInt16Ptr    = (string "uint16_t *"  <|> string "uint16_t* ")  >> pure (APtr THInt16)
thUInt16PtrPtr = (string "uint16_t **" <|> string "uint16_t** ") >> pure (APtr (APtr THInt16))

thUInt8, thUInt8Ptr, thUInt8PtrPtr :: Parser THType
thUInt8       = string "uint8_t" >> pure THInt8
thUInt8Ptr    = thPtr "uint8_t" (APtr THInt8)
thUInt8PtrPtr = (string "uint8_t **" <|> string "uint8_t** ") >> pure (APtr (APtr THInt8))

thInt64, thInt64Ptr, thInt64PtrPtr :: Parser THType
thInt64       =  string "int64_t" >> pure THInt64
thInt64Ptr    = (string "int64_t *" <|> string "int64_t* ") >> pure (APtr THInt64)
thInt64PtrPtr = (string "int64_t **" <|> string "int64_t** ") >> pure (APtr (APtr THInt64))

thInt32, thInt32Ptr, thInt32PtrPtr :: Parser THType
thInt32       =  string "int32_t" >> pure THInt32
thInt32Ptr    = (string "int32_t *" <|> string "int32_t* ") >> pure (APtr THInt32)
thInt32PtrPtr = (string "int32_t **" <|> string "int32_t** ") >> pure (APtr (APtr THInt32))

thInt16, thInt16Ptr, thInt16PtrPtr :: Parser THType
thInt16       =  string "int16_t" >> pure THInt16
thInt16Ptr    = (string "int16_t *" <|> string "int16_t* ") >> pure (APtr THInt16)
thInt16PtrPtr = (string "int16_t **" <|> string "int16_t** ") >> pure (APtr (APtr THInt16))

thInt8, thInt8Ptr, thInt8PtrPtr :: Parser THType
thInt8       = string "int8_t" >> pure THInt8
thInt8Ptr    = (string "int8_t *" <|> string "int8_t* ") >> pure (APtr THInt8)
thInt8PtrPtr = (string "int8_t **" <|> string "int8_t** ") >> pure (APtr (APtr THInt8))


thSize :: Parser THType
thSize = string "size_t" >> pure THSize

thChar, thCharPtr, thCharPtrPtr :: Parser THType
thChar       =  string "char" >> pure THChar
thCharPtr    = (string "char*" <|> string "char *") >> pure (APtr THChar)
thCharPtrPtr = (string "char**" <|> string "char **") >> pure (APtr (APtr THChar))

thShort, thShortPtr :: Parser THType
thShort    =  string "short" >> pure THShort
thShortPtr = (string "short *" <|> string "short* ") >> pure (APtr THShort)

thHalf, thHalfPtr :: Parser THType
thHalf    =  string "THHalf" >> pure THHalf
thHalfPtr = (string "THHalf *" <|> string "THHalf* ") >> pure (THHalfPtr)

thReal, thRealPtr :: Parser THType
thReal    = string "real" >> pure THReal
thRealPtr = (string "real *" <|> string "real* ") >> pure (APtr THReal)
-- TODO : clean up pointer matching


thAccReal, thAccRealPtr :: Parser THType
thAccReal    = string "accreal" >> pure THAccReal
thAccRealPtr = string "accreal *" >> pure (APtr THAccReal)

thFilePtr :: Parser THType
thFilePtr = (string "THFile *" <|> string "THFile*") >> pure (APtr THFile)

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

    -- <|> thNNStatePtr
    -- <|> thIndexTensorPtr
    -- <|> thIntegerTensorPtr

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
  (char ')') <|> (space >> char ')')
  --char ')' :: Parser Char -- TODO move this outside
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
thGenericPrefixes
  =   string "THTensor_("
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

-- thItem :: Parser (Maybe THFunction)
-- thItem = try thConstant <|> thFunctionTemplate <|> thSkip

-- NOTE: ordering is important for parsers
thParser :: CodeGenType -> Parser [Maybe THFunction]
thParser = \case
  GenericFiles  -> some
    $   try thConstant
    <|> thFunctionTemplate
    <|> thSkip
  ConcreteFiles -> some
    $   try thConstant
    <|> (thAPI >> space >> thFunctionConcrete)
    <|> thSkip
