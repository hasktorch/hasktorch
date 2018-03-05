module CodeGen.Render.Haskell
  ( renderHaskellType
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases

typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  ReturnValue   -> Just $ "IO (" <> s <> ")"
  FunctionParam -> Just s


renderHaskellType' :: TemplateType -> THType -> Text
renderHaskellType' tt =
  \case
    APtr x      -> "Ptr (" <> renderHaskellType' tt x <> ")"
    THVoid      -> "()"
    THDescBuff  -> "CTHDescBuff"

    {- NN -}
    -- THNNState       -> ("CTH" <> type2hsreal tt <> "NNState")
    -- THIndexTensor   -> "CTHIndexTensor"
    -- THIntegerTensor -> "CTHIntegerTensor"

    {- Tensor -}
    THTensor       -> ("CTH" <> type2hsreal tt <> "Tensor")
    THByteTensor   -> "CTHByteTensor"
    THCharTensor   -> "CTHCharTensor"
    THShortTensor  -> "CTHShortTensor"
    THHalfTensor   -> "CTHHalfTensor"
    THIntTensor    -> "CTHIntTensor"
    THLongTensor   -> "CTHLongTensor"
    THFloatTensor  -> "CTHFloatTensor"
    THDoubleTensor -> "CTHDoubleTensor"

    {- Storage -}
    THStorage       -> "CTH" <> type2hsreal tt <> "Storage"
    THByteStorage   -> "CTHByteStorage"
    THShortStorage  -> "CTHShortStorage"
    THIntStorage    -> "CTHIntStorage"
    THLongStorage   -> "CTHLongStorage"
    THHalfStorage   -> "CTHHalfStorage"
    THCharStorage   -> "CTHCharStorage"
    THFloatStorage  -> "CTHFloatStorage"
    THDoubleStorage -> "CTHDoubleStorage"

    {- Other -}
    THGenerator -> "CTHGenerator"  -- concrete type found in TensorMath
    THAllocator -> "CTHAllocator"
    THDouble    -> "CDouble"           -- added from TensorRandom
    THPtrDiff   -> "CPtrdiff"          -- TODO: check if it's appropriate to splice here
    THLong      -> "CLong"
    THFloat     -> "CFloat"
    THLong      -> "CLong"
    THBool      -> "CBool"
    THInt       -> "CInt"

    -- int/uint conversions, see
    -- https://www.haskell.org/onlinereport/haskell2010/haskellch8.html
    -- https://hackage.haskell.org/package/base-4.10.0.0/docs/Foreign-C-Types.html
    THUInt64    -> "CULong"
    THUInt32    -> "CUInt"
    THUInt16    -> "CUShort"
    THUInt8     -> "CBool"
    THInt64     -> "CLLong"
    THInt32     -> "Int"
    THInt16     -> "CShort"
    THInt8      -> "CSChar"
    THSize      -> "CSize"
    THChar      -> "CChar"
    THShort     -> "CShort"
    THHalf      -> "CTHHalf"
    -- FIXME: this looks like a special case which doesn't exist
    THHalfPtr   -> "Ptr CTHHalf"
    THReal      -> type2real tt
    THAccReal   -> type2accreal tt
    THFile      -> "CTHFile"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType tc tt = typeCatHelper tc . renderHaskellType' tt

{-
renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType tc tt =
  \case
    THVoid      -> case tc of { ReturnValue -> Just "IO ()" ;       FunctionParam -> Nothing }
    ATen THVoid -> case tc of { ReturnValue -> Just "IO (Ptr ())" ; FunctionParam -> Just "Ptr ()" }
    THDescBuff  -> tc `typeCatHelper` "CTHDescBuff"

    {- NN -}
    THNNStatePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2hsreal tt <> "NNState")
    THIndexTensorPtr   -> tc `typeCatHelper` "Ptr CTHIndexTensor"
    THIntegerTensorPtr -> tc `typeCatHelper` "Ptr CTHIntegerTensor"

    {- Tensor -}
    THTensorPtrPtr    -> tc `typeCatHelper` ("Ptr (Ptr CTH" <> type2hsreal tt <> "Tensor)")
    THTensorPtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2hsreal tt <> "Tensor")
    THByteTensorPtr   -> tc `typeCatHelper` "Ptr CTHByteTensor"
    THCharTensorPtr   -> tc `typeCatHelper` "Ptr CTHCharTensor"
    THShortTensorPtr  -> tc `typeCatHelper` "Ptr CTHShortTensor"
    THHalfTensorPtr   -> tc `typeCatHelper` "Ptr CTHHalfTensor"
    THIntTensorPtr    -> tc `typeCatHelper` "Ptr CTHIntTensor"
    THLongTensorPtr   -> tc `typeCatHelper` "Ptr CTHLongTensor"
    THFloatTensorPtr  -> tc `typeCatHelper` "Ptr CTHFloatTensor"
    THDoubleTensorPtr -> tc `typeCatHelper` "Ptr CTHDoubleTensor"

    {- Storage -}
    THStoragePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2hsreal tt <> "Storage")
    THByteStoragePtr   -> tc `typeCatHelper` "Ptr CTHByteStorage"
    THShortStoragePtr  -> tc `typeCatHelper` "Ptr CTHShortStorage"
    THIntStoragePtr    -> tc `typeCatHelper` "Ptr CTHIntStorage"
    THLongStoragePtr   -> tc `typeCatHelper` "Ptr CTHLongStorage"
    THHalfStoragePtr   -> tc `typeCatHelper` "Ptr CTHHalfStorage"
    THCharStoragePtr   -> tc `typeCatHelper` "Ptr CTHCharStorage"
    THFloatStoragePtr  -> tc `typeCatHelper` "Ptr CTHFloatStorage"
    THDoubleStoragePtr -> tc `typeCatHelper` "Ptr CTHDoubleStorage"

    {- Other -}
    THGeneratorPtr -> tc `typeCatHelper` "Ptr CTHGenerator"  -- concrete type found in TensorMath
    THAllocatorPtr -> tc `typeCatHelper` "CTHAllocatorPtr"
    THDoublePtr    -> tc `typeCatHelper` "Ptr CDouble"
    THDouble       -> tc `typeCatHelper` "CDouble"           -- added from TensorRandom
    THPtrDiff      -> tc `typeCatHelper` "CPtrdiff"          -- TODO: check if it's appropriate to splice here
    THLongPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CLong)"
    THLongPtr      -> tc `typeCatHelper` "Ptr CLong"
    THFloatPtr     -> tc `typeCatHelper` "Ptr CFloat"
    THFloat        -> tc `typeCatHelper` "CFloat"
    THLong         -> tc `typeCatHelper` "CLong"
    THBool         -> tc `typeCatHelper` "CBool"
    THIntPtr       -> tc `typeCatHelper` "CIntPtr"
    THInt          -> tc `typeCatHelper` "CInt"

    -- int/uint conversions, see
    -- https://www.haskell.org/onlinereport/haskell2010/haskellch8.html
    -- https://hackage.haskell.org/package/base-4.10.0.0/docs/Foreign-C-Types.html
    THUInt64       -> tc `typeCatHelper` "CULong"
    THUInt64Ptr    -> tc `typeCatHelper` "Ptr CULong"
    THUInt64PtrPtr -> tc `typeCatHelper` "Ptr (Ptr CULong)"
    THUInt32       -> tc `typeCatHelper` "CUInt"
    THUInt32Ptr    -> tc `typeCatHelper` "Ptr CUInt"
    THUInt32PtrPtr -> tc `typeCatHelper` "Ptr (Ptr CUInt)"
    THUInt16       -> tc `typeCatHelper` "CUShort"
    THUInt16Ptr    -> tc `typeCatHelper` "Ptr CUShort"
    THUInt16PtrPtr -> tc `typeCatHelper` "Ptr (Ptr CUShort)"
    THUInt8        -> tc `typeCatHelper` "CBool"
    THUInt8Ptr     -> tc `typeCatHelper` "Ptr CBool"
    THUInt8PtrPtr  -> tc `typeCatHelper` "Ptr (Ptr CBool)"
    THInt64        -> tc `typeCatHelper` "CLLong"
    THInt64Ptr     -> tc `typeCatHelper` "Ptr CLLong"
    THInt64PtrPtr  -> tc `typeCatHelper` "Ptr (Ptr CLLong)"
    THInt32        -> tc `typeCatHelper` "Int"
    THInt32Ptr     -> tc `typeCatHelper` "Ptr Int"
    THInt32PtrPtr  -> tc `typeCatHelper` "Ptr (Ptr Int)"
    THInt16        -> tc `typeCatHelper` "CShort"
    THInt16Ptr     -> tc `typeCatHelper` "Ptr CShort"
    THInt16PtrPtr  -> tc `typeCatHelper` "Ptr (Ptr CShort)"
    THInt8         -> tc `typeCatHelper` "CSChar"
    THInt8Ptr      -> tc `typeCatHelper` "Ptr CSChar"
    THInt8PtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CSChar)"
    THSize         -> tc `typeCatHelper` "CSize"
    THCharPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CChar)"
    THCharPtr      -> tc `typeCatHelper` "Ptr CChar"
    THChar         -> tc `typeCatHelper` "CChar"
    THShortPtr     -> tc `typeCatHelper` "Ptr CShort"
    THShort        -> tc `typeCatHelper` "CShort"
    THHalfPtr      -> tc `typeCatHelper` "Ptr CTHHalf"
    THHalf         -> tc `typeCatHelper` "CTHHalf"
    THRealPtr      -> tc `typeCatHelper` ("Ptr " <> type2real tt)
    THReal         -> tc `typeCatHelper` (type2real tt)
    THAccRealPtr   -> tc `typeCatHelper` ("Ptr " <> type2accreal tt)
    THAccReal      -> tc `typeCatHelper` (type2accreal tt)
    THFilePtr      -> tc `typeCatHelper` "Ptr CTHFile"
-}
