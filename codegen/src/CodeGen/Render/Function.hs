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
  [ hsname
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

-- | render a haskell function name.
mkHsname :: LibType -> CodeGenType -> SigType -> Text -> Text
mkHsname lt cgt st funname =
  ffiPrefix st <> funname


-- | Render a single function signature.
renderSig
  :: SigType
  -> LibType
  -> CodeGenType
  -> FilePath
  -> TemplateType
  -> ModuleSuffix
  -> FileSuffix
  -> (Text, Parsable, [Arg])
  -> Text
renderSig t lt cgt headerFile tt ms fs (name, retType, args) =
  T.intercalate "\n"
    [ comment lt t hsname args retType
    , foreignCall cname headerFile
    , implementation
    ]
 where
  cname, hsname :: Text
  cname = mkCname t lt ms tt cgt name
  hsname = mkHsname lt cgt t name

  implementation :: Text
  implementation =
    case (lt, cgt, fs) of
      -- NOTE: TH and THC functions differ in the THC State. TH does have a concept of THState, which
      -- is unused. Here we render some function aliases which will allow us to maintain unified
      -- backpack signatures.
      --
      -- NOTE2: In the event that we render generic functions from the TH
      -- library which _does not include THTensorRandom_, we want to post-fix these names with a @_@
      -- and use the alias to match the backpack signatures.
      (TH, GenericFiles, "TensorRandom") -> "  " <> (haskellSig lt hsname t tt args retType)
      (TH, GenericFiles, _) ->
        T.intercalate "\n"
          [ "  " <> (haskellSig lt (mkAliasRefName hsname) t tt args retType)
          , ""
          , "-- | alias of " <> mkAliasRefName hsname <> " with unused argument (for CTHState) to unify backpack signatures."
          -- , haskellSig TH hsname t tt ((Arg (TenType State) "cstate"):args) retType
          , hsname <> " = const " <> mkAliasRefName hsname
          ]
        where
          -- | TH only (and even then only generic TH files).
          --
          -- 'reference' implying "original haskell function" and alias implying
          -- "backpack-compatible function" as well as "c-native function"
          mkAliasRefName :: Text -> Text
          mkAliasRefName = (<> "_")


      _ -> "  " <> (haskellSig lt hsname t tt args retType)


