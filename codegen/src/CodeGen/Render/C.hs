module CodeGen.Render.C
  ( render
  , renderTenType
  , renderCType
  , renderNNType
  ) where

import CodeGen.Prelude
import CodeGen.Types

render :: LibType -> Parsable -> Text
render lt =
  \case
    -- special pointer cases
    Ptr x -> render lt x <> " *"
    TenType x -> renderTenType lt x
    -- NNType x -> renderNNType lt x
    CType x -> renderCType x


renderTenType :: LibType -> TenType -> Text
renderTenType lt = \case
  Real    -> "real"
  AccReal -> "accreal"
  rest -> prefix rest <> tshow rest
 where
  prefix :: TenType -> Text
  prefix t = if lt == THC && isConcreteCudaPrefixed t then "THCuda" else tshow lt


renderCType :: CType -> Text
renderCType = \case
  CUInt64  -> "uint64_t"
  CUInt32  -> "uint32_t"
  CUInt16  -> "uint16_t"
  CUInt8   -> "uint8_t"

  CInt64   -> "int64_t"
  CInt32   -> "int32_t"
  CInt16   -> "int16_t"
  CInt8    -> "int8_t"
  CInt     -> "int"

  CSize    -> "size_t"
  CLong    -> "long"
  CChar    -> "char"
  CShort   -> "short"
  CFloat   -> "float"
  CDouble  -> "double"
  CPtrdiff -> "ptrdiff_t"
  CVoid    -> "void"
  CBool    -> "bool"


-- FIXME: get back to this when THC is finished
renderNNType :: LibType -> NNType -> Text
renderNNType lt nt = tshow lt <> tshow nt

