module CodeGen.Render.Haskell
  ( render
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases (type2hsreal, type2real, type2accreal)
import qualified Data.Text as T


render :: TypeCategory -> TemplateType -> Parsable -> Maybe Text
render tc tt = typeCatHelper tc . renderParsable tt


typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  FunctionParam -> Just s
  ReturnValue   ->
    case T.take 3 s of
     "()"  -> Just   "IO ()"
     "Ptr" -> Just $ "IO (" <> s <> ")"
     _     -> Just $ "IO " <> s <> ""


renderParsable :: TemplateType -> Parsable -> Text
renderParsable tt =
  \case
    -- special cases
    TenType desc@(Pair (DescBuff, _)) -> "Ptr " <> renderTenType tt desc

    Ptr (Ptr x) -> "Ptr (Ptr " <> renderParsable tt x <> ")"
    Ptr x -> "Ptr " <> renderParsable tt x
    -- Raw DescBuffs need to be wrapped in a pointer for marshalling
    TenType x -> renderTenType tt x
    -- NNType x -> renderNNType lt tt x
    CType x -> renderCType x


renderTenType :: TemplateType -> TenType -> Text
renderTenType tt = \case
  Pair (Tensor,  lt) -> c <> prefix lt True <> type2hsreal tt <> "Tensor"
  Pair (Storage, lt) -> c <> tshow lt       <> type2hsreal tt <> "Storage"
  Pair (Real,    lt) -> type2real lt tt
  Pair (AccReal, lt) -> type2accreal lt tt
  r@(Pair (rtt, lt)) -> c <> prefix lt (isConcreteCudaPrefixed r) <> tshow rtt
 where
  c = "C'"


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

{-
-- FIXME: get back to this when THC is finished
renderNNType :: LibType -> TemplateType -> NNType -> Text
renderNNType _ _ = \case
  IndexTensor   -> undefined
  IntegerTensor -> undefined
-}

