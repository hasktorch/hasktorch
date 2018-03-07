module CodeGen.Render.Haskell
  ( render
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases


render :: LibType -> TypeCategory -> TemplateType -> Parsable -> Maybe Text
render lt tc tt = typeCatHelper tc . renderParsable lt tt


typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  ReturnValue   -> Just $ "IO (" <> s <> ")"
  FunctionParam -> Just s


hsPrefix :: LibType -> Text
hsPrefix lt = "C" <> tshow lt


renderParsable :: LibType -> TemplateType -> Parsable -> Text
renderParsable lt tt =
  \case
    -- special pointer cases
    Ptr (TenType Allocator) -> "CTHAllocatorPtr"
    Ptr x -> "Ptr (" <> renderParsable lt tt x <> ")"
    TenType x -> renderTenType lt tt x
    NNType x -> renderNNType lt tt x
    CType x -> renderCType x


renderTenType :: LibType -> TemplateType -> TenType -> Text
renderTenType lt tt = \case
  Tensor  -> hsPrefix lt <> type2hsreal tt <> "Tensor"
  Storage -> hsPrefix lt <> type2hsreal tt <> "Storage"
  Real    -> type2real tt
  AccReal -> type2accreal tt
  rest    -> hsPrefix lt <> tshow rest


renderCType :: CType -> Text
renderCType = \case
  CVoid -> "()"
  -- int/uint conversions, see
  -- https://www.haskell.org/onlinereport/haskell2010/haskellch8.html
  -- https://hackage.haskell.org/package/base-4.10.0.0/docs/Foreign-C-Types.html
  CUInt64 -> "CULong"
  CUInt32 -> "CUInt"
  CUInt16 -> "CUShort"
  CUInt8  -> "CUChar"
  CInt64  -> "CLLong"
  CInt32  -> "CInt"
  CInt16  -> "CShort"
  CInt8   -> "CSChar"
  rest    -> tshow rest


-- FIXME: get back to this when THC is finished
renderNNType :: LibType -> TemplateType -> NNType -> Text
renderNNType _ _ = \case
  NNState       -> undefined
  IndexTensor   -> undefined
  IntegerTensor -> undefined


