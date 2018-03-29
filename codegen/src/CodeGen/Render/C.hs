module CodeGen.Render.C
  ( renderCType
  , type2real
  , type2accreal
  ) where

import CodeGen.Prelude
import CodeGen.Types
import ConditionalCases

renderCType :: THType -> Text
renderCType = \case
  THVoid             -> "void"
  THBool             -> "bool"
  THDescBuff         -> "THDescBuff"
  THNNStatePtr       -> "THNNState *"
  THTensorPtr        -> "THTensor *"
  THIntegerTensorPtr -> "THIntegerTensor *"
  THIndexTensorPtr   -> "THIndexTensor *"
  THTensorPtrPtr     -> "THTensor **"
  THByteTensorPtr    -> "THByteTensor *"
  THLongTensorPtr    -> "THLongTensor *"
  THDoubleTensorPtr  -> "THDoubleTensor *"
  THFloatTensorPtr   -> "THFloatTensor *"
  THGeneratorPtr     -> "THGenerator *"
  THStoragePtr       -> "THStorage *"
  THCharStoragePtr   -> "THCharStorage *"
  THLongStoragePtr   -> "THLongStorage *"
  THPtrDiff          -> "ptrdiff_t"
  THLongPtrPtr       -> "long **"
  THLongPtr          -> "long *"
  THLong             -> "long"
  THIntPtr           -> "int *"
  THInt              -> "int"

  THUInt64           -> "uint64_t"
  THUInt64Ptr        -> "uint64_t *"
  THUInt64PtrPtr     -> "uint64_t **"
  THUInt32           -> "uint32_t"
  THUInt32Ptr        -> "uint32_t *"
  THUInt32PtrPtr     -> "uint32_t **"
  THUInt16           -> "uint16_t"
  THUInt16Ptr        -> "uint16_t *"
  THUInt16PtrPtr     -> "uint16_t **"
  THUInt8            -> "uint8_t"
  THUInt8Ptr         -> "uint8_t *"
  THUInt8PtrPtr      -> "uint8_t **"

  THInt64            -> "int64_t"
  THInt64Ptr         -> "int64_t *"
  THInt64PtrPtr      -> "int64_t **"
  THInt32            -> "int32_t"
  THInt32Ptr         -> "int32_t *"
  THInt32PtrPtr      -> "int32_t **"
  THInt16            -> "int16_t"
  THInt16Ptr         -> "int16_t *"
  THInt16PtrPtr      -> "int16_t **"
  THInt8             -> "int8_t"
  THInt8Ptr          -> "int8_t *"
  THInt8PtrPtr       -> "int8_t **"
  THSize             -> "size_t"
  THCharPtr          -> "char *"
  THChar             -> "char"
  THShort            -> "short"
  THHalf             -> "THHalf"
  THHalfPtr          -> "THHalfPtr"
  THFloat            -> "float"
  THDouble           -> "double"
  THRealPtr          -> "real *"
  THReal             -> "real"
  THAccRealPtr       -> "accreal *"
  THAccReal          -> "accreal"
  THFilePtr          -> "THFile *"
  s -> error (show s <> " is unaccounted for") -- TODO : make this total

-- See header files "#define real [X]"
type2real :: TemplateType -> Text
type2real t = case signatureAliases t of
  Just (_, CReal _ (CRep t), _, _) -> t
  Nothing -> impossible "TemplateType is concrete and should not have been called"

-- See header files "#define accreal [X]"
type2accreal :: TemplateType -> Text
type2accreal t = case signatureAliases t of
  Just (_, _, CAccReal _ (CRep t), _) -> t
  Nothing -> impossible "TemplateType is concrete and should not have been called"


