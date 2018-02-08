module CodeGen.Render.Function
  ( renderFunSig
  , renderFunPtrSig
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Render.C (renderCType)
import CodeGen.Render.Haskell (renderHaskellType)
import qualified Data.Text as T

data SigType
  = IsFun
  | IsFunPtr
  deriving (Eq, Ord, Show)

ffiPrefix :: SigType -> Text
ffiPrefix = \case
  IsFun    -> "c_"
  IsFunPtr -> "p_"

isPtr :: SigType -> Bool
isPtr f = f == IsFunPtr

renderCName :: Text -> Text -> Text
renderCName prefix name = prefix <> "_" <> name

-- | Render a single function signature.
renderSig :: SigType -> Bool -> Text -> FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderSig t isTemplate prefix headerFile modTypeTemplate (name, retType, args) =
  trace (if "THShortStorage" == prefix && "THStorage.h" == headerFile then show (name, thArgType <$> args, retArrow) else "") $
  T.intercalate "\n"
    [ comment, foreignCall t, haskellSig t ]
 where
  comment :: Text
  comment = T.intercalate " "
    $  [ "-- |" , ffiPrefix t <> name , ":", if isPtr t then "Pointer to function :" else "" ]
    <> (map thArgName args)
    <> (if null args then [] else ["->"])
    <> [renderCType retType]

  cName :: Text
  cName = if isTemplate then renderCName prefix name else name

  foreignCall :: SigType -> Text
  foreignCall = \case
    IsFun    -> T.intercalate "\"" [ "foreign import ccall ", T.pack headerFile <> " "  <> cName, "" ]
    IsFunPtr -> T.intercalate "\"" [ "foreign import ccall ", T.pack headerFile <> " &" <> cName, "" ]

  haskellSig :: SigType -> Text
  haskellSig = \case
    IsFun -> T.intercalate ""
      [ "  c_" <> name <> " :: " <> T.intercalate " -> " typeSignature, retArrow ]
    IsFunPtr -> T.intercalate ""
      [ "  p_" <> name <> " :: FunPtr (" <> T.intercalate " -> " typeSignature, retArrow, ")" ]

  typeSignature :: [Text]
  typeSignature = mapMaybe (renderHaskellType FunctionParam modTypeTemplate . thArgType) args

  retArrow :: Text
  retArrow = case renderHaskellType ReturnValue modTypeTemplate retType of
    Nothing  -> ""
    Just ret -> if null typeSignature then ret else (" -> " <> ret)


-- | Render a single function signature.
renderFunSig :: Bool -> Text -> FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunSig = renderSig IsFun

-- | Render function pointer signature
renderFunPtrSig :: Bool -> Text -> FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunPtrSig = renderSig IsFunPtr


