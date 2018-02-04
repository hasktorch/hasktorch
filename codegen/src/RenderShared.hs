{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE NamedFieldPuns #-}
module RenderShared (
  makeModule,
  renderCType,
  type2SpliceReal,
  type2real,
  type2accreal,
  realtype2Haskell,
  accrealtype2Haskell,

  renderCHeaderFile,
  parseFile,
  cleanList
  ) where

import Data.List (nub)
import Data.Maybe (fromJust, catMaybes, isJust, Maybe)
import Data.Either (either)
import Data.Monoid ((<>))
import Data.Text (Text)
import Data.Void
import Text.Megaparsec
import Text.Show.Pretty
import Debug.Trace
import Prelude
import qualified Data.Text as T
import qualified Prelude as P

import CodeGenTypes
import CodeGenParse
import ConditionalCases

makeModule :: Text -> Bool -> FilePath -> Text -> Text -> TemplateType -> [THFunction] -> HModule
makeModule outDir isTemplate modHeader modSuffix modFileSuffix typeTemplate bindings
  = HModule
  { modHeader = modHeader
  , modPrefix = "TH"
  , modTypeTemplate = typeTemplate
  , modSuffix = modSuffix
  , modFileSuffix = modFileSuffix
  , modExtensions = ["ForeignFunctionInterface"]
  , modImports = ["Foreign", "Foreign.C.Types", "THTypes", "Data.Word", "Data.Int"]
  , modTypeDefs = []
  , modBindings = bindings
  , modOutDir = outDir
  , modIsTemplate = isTemplate
  }

-- TODO : make this total
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

-- ----------------------------------------
-- helper data and functions for templating
-- ----------------------------------------

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal = \case
  GenByte    -> "Byte"
  GenChar    -> "Byte"
  GenDouble  -> "Double"
  GenFloat   -> "Float"
  GenHalf    -> "Half"
  GenInt     -> "Int"
  GenLong    -> "Long"
  GenShort   -> "Short"
  GenNothing -> ""

-- See header files "#define real [X]"
type2real :: TemplateType -> Text
type2real = \case
  GenByte   -> "unsigned char"
  GenChar   -> "char"
  GenDouble -> "double"
  GenFloat  -> "float"
  GenHalf   -> "THHalf"
  GenInt    -> "int"
  GenLong   -> "long"
  GenShort  -> "short"

-- See header files "#define accreal [X]"
type2accreal :: TemplateType -> Text
type2accreal = \case
  GenByte   -> "long"
  GenChar   -> "long"
  GenDouble -> "double"
  GenFloat  -> "double"
  GenHalf   -> "float"
  GenInt    -> "long"
  GenLong   -> "long"
  GenShort  -> "long"

realtype2Haskell :: TemplateType -> Text
realtype2Haskell = \case
  GenByte   -> "CUChar"
  GenChar   -> "CChar"
  GenDouble -> "CDouble"
  GenFloat  -> "CFloat"
  GenHalf   -> "THHalf"
  GenInt    -> "CInt"
  GenLong   -> "CLong"
  GenShort  -> "CShort"

accrealtype2Haskell :: TemplateType -> Text
accrealtype2Haskell = \case
  GenByte   -> "CLong"
  GenChar   -> "CLong"
  GenDouble -> "CDouble"
  GenFloat  -> "CDouble"
  GenHalf   -> "CFloat"
  GenInt    -> "CLong"
  GenLong   -> "CLong"
  GenShort  -> "CLong"

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType tc tt =
  \case
    THVoid     -> case tc of { ReturnValue -> Just "IO ()" ; FunctionParam -> Nothing }
    THVoidPtr  -> Just "Ptr ()"
    THDescBuff -> Just "CTHDescBuff"

    {- NN -}
    THNNStatePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "NNState")
    THIndexTensorPtr   -> tc `typeCatHelper` "Ptr CTHIndexTensor"
    THIntegerTensorPtr -> tc `typeCatHelper` "Ptr CTHIntegerTensor"

    {- Tensor -}
    THTensorPtrPtr    -> tc `typeCatHelper` ("Ptr (Ptr CTH" <> type2SpliceReal tt <> "Tensor)")
    THTensorPtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "Tensor")
    THByteTensorPtr   -> tc `typeCatHelper` "Ptr CTHByteTensor"
    THCharTensorPtr   -> tc `typeCatHelper` "Ptr CTHCharTensor"
    THShortTensorPtr  -> tc `typeCatHelper` "Ptr CTHShortTensor"
    THHalfTensorPtr   -> tc `typeCatHelper` "Ptr CTHHalfTensor"
    THIntTensorPtr    -> tc `typeCatHelper` "Ptr CTHIntTensor"
    THLongTensorPtr   -> tc `typeCatHelper` "Ptr CTHLongTensor"
    THFloatTensorPtr  -> tc `typeCatHelper` "Ptr CTHFloatTensor"
    THDoubleTensorPtr -> tc `typeCatHelper` "Ptr CTHDoubleTensor"

    {- Storage -}
    THStoragePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "Storage")
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
    THDouble       -> Just "CDouble"                         -- added from TensorRandom
    THPtrDiff      -> Just "CPtrdiff"                        -- TODO: check if it's appropriate to splice here
    THLongPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CLong)"
    THLongPtr      -> tc `typeCatHelper` "Ptr CLong"
    THFloatPtr     -> tc `typeCatHelper` "Ptr CFloat"
    THFloat        -> Just "CFloat"
    THLong         -> Just "CLong"
    THBool         -> Just "CBool"
    THIntPtr       -> tc `typeCatHelper` "CIntPtr"
    THInt          -> Just "CInt"

    -- int/uint conversions, see
    -- https://www.haskell.org/onlinereport/haskell2010/haskellch8.html
    -- https://hackage.haskell.org/package/base-4.10.0.0/docs/Foreign-C-Types.html
    THUInt64       -> Just "CULong"
    THUInt64Ptr    -> Just "Ptr CULong"
    THUInt64PtrPtr -> Just "Ptr (Ptr CULong)"
    THUInt32       -> Just "CUInt"
    THUInt32Ptr    -> Just "Ptr CUInt"
    THUInt32PtrPtr -> Just "Ptr (Ptr CUInt)"
    THUInt16       -> Just "CUShort"
    THUInt16Ptr    -> Just "Ptr CUShort"
    THUInt16PtrPtr -> Just "Ptr (Ptr CUShort)"
    THUInt8        -> Just "CBool"
    THUInt8Ptr     -> Just "Ptr CBool"
    THUInt8PtrPtr  -> Just "Ptr (Ptr CBool)"
    THInt64        -> Just "CLLong"
    THInt64Ptr     -> Just "Ptr CLLong"
    THInt64PtrPtr  -> Just "Ptr (Ptr CLLong)"
    THInt32        -> Just "Int"
    THInt32Ptr     -> Just "Ptr Int"
    THInt32PtrPtr  -> Just "Ptr (Ptr Int)"
    THInt16        -> Just "CShort"
    THInt16Ptr     -> Just "Ptr CShort"
    THInt16PtrPtr  -> Just "Ptr (Ptr CShort)"
    THInt8         -> Just "CSChar"
    THInt8Ptr      -> Just "Ptr CSChar"
    THInt8PtrPtr   -> Just "Ptr (Ptr CSChar)"
    THSize         -> Just "CSize"
    THCharPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CChar)"
    THCharPtr      -> tc `typeCatHelper` "Ptr CChar"
    THChar         -> Just "CChar"
    THShortPtr     -> tc `typeCatHelper` "Ptr CShort"
    THShort        -> Just "CShort"
    THHalfPtr      -> tc `typeCatHelper` "Ptr CTHHalf"
    THHalf         -> Just "CTHHalf"
    THRealPtr      -> tc `typeCatHelper` ("Ptr " <> realtype2Haskell tt)
    THReal         -> Just (realtype2Haskell tt)
    THAccRealPtr   -> tc `typeCatHelper` ("Ptr " <> accrealtype2Haskell tt)
    THAccReal      -> Just (accrealtype2Haskell tt)
    THFilePtr      -> tc `typeCatHelper` "Ptr CTHFile"

typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  ReturnValue   -> Just $ "IO (" <> s <> ")"
  FunctionParam -> Just s

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> " #-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions =
  T.intercalate "\n" (renderExtension <$> extensions) <> "\n\n"

renderModule :: HModule -> Text
renderModule moduleSpec = "module " <> renderModuleName moduleSpec

renderExports :: [Text] -> Text
renderExports exports = T.intercalate "\n"
  [ ""
  , "  ( " <> T.intercalate "\n  , " exports
  , "  ) where"
  , ""
  , ""
  ]

renderImports :: [Text] -> Text
renderImports imports = T.intercalate "\n" (("import " <>) <$> imports) <> "\n\n"

renderFunName :: Text -> Text -> Text
renderFunName prefix name = prefix <> "_" <> name

-- |Render a single function signature.
renderFunSig :: FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunSig headerFile modTypeTemplate (name, retType, args) = T.intercalate ""
  [ "-- |c_" <> name <> " : " <> T.intercalate " " nameSignature <> " -> " <> (renderCType retType) <> "\n"
   --   <> "foreign import ccall unsafe \"" <> T.pack headerFile <> " " <> name <> "\"\n"
  , "foreign import ccall \"" <> T.pack headerFile <> " " <> name <> "\"\n"
  , "  c_" <> name <> " :: " <> T.intercalate " -> " typeSignatureClean
    -- TODO : fromJust shouldn't fail but still clean this up so it's not unsafe
  , retArrow <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType)
  ]
 where
  typeVals = thArgType <$> args
  typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
  typeSignatureClean = catMaybes typeSignature
  numArgs = P.length typeSignatureClean
  retArrow = if numArgs == 0 then "" else " -> "
  nameSignature = thArgName <$> args

-- |Render function pointer signature
renderFunPtrSig :: FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunPtrSig headerFile modTypeTemplate (name, retType, args) = T.intercalate ""
  [ "-- |p_" <> name <> " : Pointer to function : "

  , T.intercalate " " nameSignature <> " -> " <> renderCType retType <> "\n"

  , "foreign import ccall \"" <> T.pack headerFile <> " &" <> name <> "\"\n"

  , "  p_" <> name <> " :: FunPtr (" <> T.intercalate " -> " typeSignatureClean
   -- TODO : fromJust shouldn't fail but still clean this up so it's not unsafe
  , retArrow <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType)

  , ")"
  ]
 where
  typeVals = thArgType <$> args
  typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
  typeSignatureClean = catMaybes typeSignature
  numArgs = P.length typeSignatureClean
  retArrow = if numArgs == 0 then "" else " -> "
  nameSignature = thArgName <$> args

-- TODO clean up redundancy of valid functions vs. functions in moduleSpec
renderFunctions :: HModule -> [THFunction] -> Text
renderFunctions m validFunctions =
  T.intercalate "\n\n"
    $  (renderFunSig'    <$> triple)
    <> (renderFunPtrSig' <$> triple)
 where
  renderFunSig'    = renderFunSig    (modHeader m) (modTypeTemplate m)
  renderFunPtrSig' = renderFunPtrSig (modHeader m) (modTypeTemplate m)

  modulePrefix :: Text
  modulePrefix = modPrefix m <> type2SpliceReal (modTypeTemplate m) <> modSuffix m <> "_"

  triple :: [(Text, THType, [THArg])]
  triple = go <$> validFunctions
    where
      go :: THFunction -> (Text, THType, [THArg])
      go f = (funName f, funReturn f, funArgs f)

-- | Check for conditional templating of functions and filter function list
checkList :: [THFunction] -> TemplateType -> [THFunction]
checkList fList templateType = P.filter ((checkFunction templateType) . funName) fList

renderAll :: HModule -> Text
renderAll spec
  =  trace (show (prefix, splice)) $ renderExtensions (modExtensions spec)
  <> renderModule spec
  <> renderExports exportFunctions
  <> renderImports (modImports spec)
  <> renderFunctions spec validFunctions
  where
    prefix, splice :: Text
    prefix = makePrefix . type2SpliceReal . modTypeTemplate $ spec
    splice = {-modPrefix spec <> -} type2SpliceReal (modTypeTemplate spec) <> modSuffix spec

    validFunctions :: [THFunction]
    validFunctions = checkList (modBindings spec) (modTypeTemplate spec)

    fun2name :: Text -> THFunction -> Text
    fun2name p = (\f -> p <> "_" <> f) . funName

    exportFunctions :: [Text]
    exportFunctions =
      if modIsTemplate spec
      then (fmap (fun2name ("c_" <> splice)) validFunctions)
        <> (fmap (fun2name ("p_" <> splice)) validFunctions)
      else (fmap (fun2name "c") validFunctions)
        <> (fmap (fun2name "p") validFunctions)

renderCHeaderFile :: TemplateType -> [THFunction] -> (TemplateType -> [THFunction] -> HModule) -> IO ()
renderCHeaderFile templateType parsedBindings makeConfig = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile (outDir ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
 where
  modSpec :: HModule
  modSpec = makeConfig templateType parsedBindings

  filename :: Text
  filename = renderModuleName modSpec <> ".hs"

  outDir :: String
  outDir = T.unpack (modOutDir modSpec)

renderModuleName :: HModule -> Text
renderModuleName HModule{modPrefix, modTypeTemplate, modFileSuffix}
  = modPrefix <> (type2SpliceReal modTypeTemplate) <> modFileSuffix

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- |Remove If list was returned, extract non-Nothing values, o/w empty list
cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList = either (const []) catMaybes

parseFile :: String -> IO [THFunction]
parseFile file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thParseGeneric file
  pure $ cleanList res
  where
    parseFromFile :: Parser [Maybe THFunction] -> String -> IO (Either (ParseError Char Void) [Maybe THFunction])
    parseFromFile p file = runParser p file <$> readFile file
