module CodeGen.Render.Function
  ( renderSig
  , SigType(..)
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases (type2hsreal)
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

comment :: SigType -> Text -> [THArg] -> THType -> Text
comment t hsname args retType = T.intercalate " "
  $  [ "-- |" , hsname , ":", if isPtr t then "Pointer to function :" else "" ]
  <> map thArgName args
  <> (if null args then [] else ["->"])
  <> [renderCType retType]

foreignCall :: Text -> FilePath -> SigType -> Text
foreignCall cname headerFile st = T.intercalate "\""
  [ "foreign import ccall "
  , T.pack headerFile <> cname
  , ""
  ]

haskellSig :: Text -> SigType -> TemplateType -> [THArg] -> THType -> Text
haskellSig hsname st tt args retType = T.intercalate ""
  [ "  " <> hsname
  , " :: "
  , if isPtr st then "FunPtr (" else ""
  , T.intercalate " -> " typeSignature, retArrow
  , if isPtr st then ")" else ""
  ]
 where
  typeSignature :: [Text]
  typeSignature = case args of
    [THArg THVoid _] -> []
    args' -> mapMaybe (renderHaskellType FunctionParam tt . thArgType) args'

  retArrow :: Text
  retArrow = case renderHaskellType ReturnValue tt retType of
    Nothing  -> ""
    Just ret -> if null typeSignature then ret else (" -> " <> ret)


mkCname :: SigType -> LibType -> ModuleSuffix -> TemplateType -> CodeGenType -> Text -> Text
mkCname st lt ms tt cgt funname
  = (if isPtr st then " &" else " ")
  <> identifier
  <> funname
 where
  identifier :: Text
  identifier = case cgt of
    GenericFiles -> tshow lt <> type2hsreal tt <> textSuffix ms <> "_"
    ConcreteFiles -> ""

mkHsname :: SigType -> Text -> Text
mkHsname st funname = ffiPrefix st <> funname


-- | Render a single function signature.
renderSig
  :: SigType
  -> CodeGenType
  -> FilePath
  -> TemplateType
  -> ModuleSuffix
  -> (Text, THType, [THArg])
  -> Text
renderSig t cgt headerFile tt ms (name, retType, args) =
    T.intercalate "\n"
      [ comment t hsname args retType
      , foreignCall cname headerFile t
      , haskellSig hsname t tt args retType
      ]
 where
  cname = mkCname t TH ms tt cgt name
  hsname = mkHsname t name


