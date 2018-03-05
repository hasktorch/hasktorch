module CodeGen.Render.C
  ( renderCType
  , type2real
  , type2accreal
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases

renderCType :: THType -> Text
renderCType = \case
  APtr x     -> renderCType x <> " *"
  THVoid     -> "void"
  THBool     -> "bool"
  THDescBuff -> "THDescBuff"

  -- THNNState       -> "THNNState"
  -- THIntegerTensor -> "THIntegerTensor"
  -- THIndexTensor   -> "THIndexTensor"
  THTensor        -> "THTensor"
  THTensor        -> "THTensor"
  THByteTensor    -> "THByteTensor"
  THLongTensor    -> "THLongTensor"
  THDoubleTensor  -> "THDoubleTensor"
  THFloatTensor   -> "THFloatTensor"
  THGenerator     -> "THGenerator"
  THStorage       -> "THStorage"
  THCharStorage   -> "THCharStorage"
  THLongStorage   -> "THLongStorage"

  THPtrDiff -> "ptrdiff_t"
  THLong    -> "long"
  THInt     -> "int"

  THUInt64  -> "uint64_t"
  THUInt32  -> "uint32_t"
  THUInt16  -> "uint16_t"
  THUInt8   -> "uint8_t"

  THInt64   -> "int64_t"
  THInt32   -> "int32_t"
  THInt16   -> "int16_t"
  THInt8    -> "int8_t"
  THSize    -> "size_t"
  THChar    -> "char"
  THShort   -> "short"
  THHalf    -> "THHalf"
  THHalfPtr -> "THHalfPtr"
  THFloat   -> "float"
  THDouble  -> "double"
  THReal    -> "real"
  THAccReal -> "accreal"
  THFile    -> "THFile"
  s -> error (show s <> " is unaccounted for") -- TODO : make this total


