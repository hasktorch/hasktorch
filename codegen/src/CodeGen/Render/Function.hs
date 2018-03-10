module CodeGen.Render.Function
  ( renderSig
  , SigType(..)
  ) where

import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Parse.Cases (type2hsreal)
import qualified CodeGen.Render.C as C (render)
import qualified CodeGen.Render.Haskell as Hs (render)
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

comment :: LibType -> SigType -> Text -> [Arg] -> Parsable -> Text
comment lt t hsname args retType = T.intercalate " "
  $  [ "-- |" , hsname , ":", if isPtr t then "Pointer to function :" else "" ]
  <> map argName args
  <> (if null args then [] else ["->"])
  <> [C.render lt retType]

foreignCall :: Text -> FilePath -> Text
foreignCall cname headerFile = T.intercalate "\""
  [ "foreign import ccall "
  , T.pack headerFile <> cname
  , ""
  ]

haskellSig :: LibType -> Text -> SigType -> TemplateType -> [Arg] -> Parsable -> Text
haskellSig lt hsname st tt args retType = T.intercalate ""
  [ "  " <> hsname
  , " :: "
  , if isPtr st then "FunPtr (" else ""
  , T.intercalate " -> " typeSignature, retArrow
  , if isPtr st then ")" else ""
  ]
 where
  typeSignature :: [Text]
  typeSignature = case args of
    [Arg (CType CVoid) _] -> []
    args' -> mapMaybe (Hs.render lt FunctionParam tt . argType) args'

  retArrow :: Text
  retArrow = case Hs.render lt ReturnValue tt retType of
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
  -> LibType
  -> CodeGenType
  -> FilePath
  -> TemplateType
  -> ModuleSuffix
  -> (Text, Parsable, [Arg])
  -> Text
renderSig t lt cgt headerFile tt ms (name, retType, args) =
    T.intercalate "\n"
      [ comment lt t hsname args retType
      , foreignCall cname headerFile
      , haskellSig lt hsname t tt args retType
      ]
 where
  cname = mkCname t lt ms tt cgt name
  hsname = mkHsname t name


