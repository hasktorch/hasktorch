module CodeGen.Render.Haskell
  ( render
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases
import qualified Data.Text as T


render :: LibType -> TypeCategory -> TemplateType -> Parsable -> Maybe Text
render lt tc tt = typeCatHelper tc . renderParsable lt tt


typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  FunctionParam -> Just s
  ReturnValue   ->
    case T.take 3 s of
     "()"  -> Just   "IO ()"
     "Ptr" -> Just $ "IO (" <> s <> ")"
     _     -> Just $ "IO " <> s <> ""


renderParsable :: LibType -> TemplateType -> Parsable -> Text
renderParsable lt tt =
  \case
    -- special cases
    TenType DescBuff -> "Ptr " <> renderTenType lt tt DescBuff

    Ptr (Ptr x) -> "Ptr (Ptr " <> renderParsable lt tt x <> ")"
    Ptr x -> "Ptr " <> renderParsable lt tt x
    -- Raw DescBuffs need to be wrapped in a pointer for marshalling
    TenType x -> renderTenType lt tt x
    -- NNType x -> renderNNType lt tt x
    CType x -> renderCType x


renderTenType :: LibType -> TemplateType -> TenType -> Text
renderTenType lt tt = \case
  Tensor  -> c <> libPrefix True <> type2hsreal tt <> "Tensor"
  Storage -> c <> tshow lt       <> type2hsreal tt <> "Storage"
  Real    -> type2real lt tt
  AccReal -> type2accreal lt tt
  rest    -> c <> libPrefix (isConcreteCudaPrefixed rest) <> tshow rest
 where
  c = "C'"
  libPrefix :: Bool -> Text
  libPrefix long =
    case lt of
      THC -> if long then "THCuda" else "THC"
      _   -> tshow lt


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

