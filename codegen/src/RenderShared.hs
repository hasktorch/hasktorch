{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

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
import Data.Maybe (fromJust, catMaybes)
import Data.Monoid ((<>))
import Prelude as P
import Data.Text
import Data.Text as T
import Data.Void
import Text.Megaparsec
import Text.Show.Pretty

import CodeGenTypes
import CodeGenParse
import ConditionalCases

makeModule ::
  Text -> Bool -> FilePath -> Text -> Text -> TemplateType -> [THFunction] -> HModule
makeModule outDir isTemplate modHeader modSuffix modFileSuffix typeTemplate bindings =
   HModule {
        modHeader = modHeader,
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = modSuffix,
        modFileSuffix = modFileSuffix,
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings,
        modOutDir = outDir,
        modIsTemplate = isTemplate
  }

-- TODO : make this total
renderCType :: THType -> Text
renderCType THVoid            = "void"
renderCType THDescBuff        = "THDescBuff"
renderCType THTensorPtr       = "THTensor *"
renderCType THTensorPtrPtr    = "THTensor **"
renderCType THByteTensorPtr   = "THByteTensor *"
renderCType THLongTensorPtr   = "THLongTensor *"
renderCType THDoubleTensorPtr = "THDoubleTensor *"
renderCType THFloatTensorPtr  = "THFloatTensor *"
renderCType THGeneratorPtr    = "THGenerator *"
renderCType THStoragePtr      = "THStorage *"
renderCType THCharStoragePtr  = "THCharStorage *"
renderCType THLongStoragePtr  = "THLongStorage *"
renderCType THPtrDiff         = "ptrdiff_t"
renderCType THLongPtrPtr         = "long **"
renderCType THLongPtr         = "long *"
renderCType THLong            = "long"
renderCType THIntPtr          = "int *"
renderCType THInt             = "int"
renderCType THSize            = "size_t"
renderCType THCharPtr         = "char *"
renderCType THChar            = "char"
renderCType THShort           = "short"
renderCType THHalf            = "THHalf"
renderCType THHalfPtr            = "THHalfPtr"
renderCType THFloat           = "float"
renderCType THDouble          = "double"
renderCType THRealPtr         = "real *"
renderCType THReal            = "real"
renderCType THAccRealPtr      = "accreal *"
renderCType THAccReal         = "accreal"
renderCType THFilePtr         = "THFile *"

-- ----------------------------------------
-- helper data and functions for templating
-- ----------------------------------------

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal GenByte   = "Byte"
type2SpliceReal GenChar   = "Byte"
type2SpliceReal GenDouble = "Double"
type2SpliceReal GenFloat  = "Float"
type2SpliceReal GenHalf   = "Half"
type2SpliceReal GenInt    = "Int"
type2SpliceReal GenLong   = "Long"
type2SpliceReal GenShort  = "Short"
type2SpliceReal GenNothing = ""

-- See header files "#define real [X]"
type2real :: TemplateType -> Text
type2real GenByte   = "unsigned char"
type2real GenChar   = "char"
type2real GenDouble = "double"
type2real GenFloat  = "float"
type2real GenHalf   = "THHalf"
type2real GenInt    = "int"
type2real GenLong   = "long"
type2real GenShort  = "short"

-- See header files "#define accreal [X]"
type2accreal :: TemplateType -> Text
type2accreal GenByte   = "long"
type2accreal GenChar   = "long"
type2accreal GenDouble = "double"
type2accreal GenFloat  = "double"
type2accreal GenHalf   = "float"
type2accreal GenInt    = "long"
type2accreal GenLong   = "long"
type2accreal GenShort  = "long"

realtype2Haskell :: TemplateType -> Text
realtype2Haskell GenByte   = "CUChar"
realtype2Haskell GenChar   = "CChar"
realtype2Haskell GenDouble = "CDouble"
realtype2Haskell GenFloat  = "CFloat"
realtype2Haskell GenHalf   = "THHalf"
realtype2Haskell GenInt    = "CInt"
realtype2Haskell GenLong   = "CLong"
realtype2Haskell GenShort  = "CShort"

accrealtype2Haskell :: TemplateType -> Text
accrealtype2Haskell GenByte   = "CLong"
accrealtype2Haskell GenChar   = "CLong"
accrealtype2Haskell GenDouble = "CDouble"
accrealtype2Haskell GenFloat  = "CDouble"
accrealtype2Haskell GenHalf   = "CFloat"
accrealtype2Haskell GenInt    = "CLong"
accrealtype2Haskell GenLong   = "CLong"
accrealtype2Haskell GenShort  = "CLong"

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text

renderHaskellType _ templateType THVoidPtr = Just "Ptr ()"

renderHaskellType typeCat templateType THVoid =
  case typeCat of
    ReturnValue -> Just "IO ()"
    FunctionParam -> Nothing

renderHaskellType _ _ THDescBuff = Just "CTHDescBuff"

{- Tensor -}

renderHaskellType typeCat templateType THTensorPtrPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor))"
  FunctionParam -> Just $ "Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType typeCat templateType THTensorPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"
  FunctionParam -> Just $ "(Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType typeCat _ THByteTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHByteTensor)"
  FunctionParam -> Just "Ptr CTHByteTensor"

renderHaskellType typeCat _ THCharTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHCharTensor)"
  FunctionParam -> Just "Ptr CTHCharTensor"

renderHaskellType typeCat _ THShortTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHShortTensor)"
  FunctionParam -> Just "Ptr CTHShortTensor"

renderHaskellType typeCat _ THHalfTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHHalfTensor)"
  FunctionParam -> Just "Ptr CTHHalfTensor"

renderHaskellType typeCat _ THIntTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHIntTensor)"
  FunctionParam -> Just "Ptr CTHIntTensor"

renderHaskellType typeCat _ THLongTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHLongTensor)"
  FunctionParam -> Just "Ptr CTHLongTensor"

renderHaskellType typeCat _ THFloatTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHFloatTensor)"
  FunctionParam -> Just "Ptr CTHFloatTensor"

renderHaskellType typeCat _ THDoubleTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHDoubleTensor)"
  FunctionParam -> Just "Ptr CTHDoubleTensor"

{- Storage -}

renderHaskellType typeCat templateType THStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTH" <> type2SpliceReal templateType <> "Storage)"
  FunctionParam -> Just $ "Ptr CTH" <> type2SpliceReal templateType <> "Storage"

renderHaskellType typeCat _ THByteStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHByteStorage)"
  FunctionParam -> Just $ "Ptr CTHByteStorage"

renderHaskellType typeCat _ THShortStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHShortStorage)"
  FunctionParam -> Just $ "Ptr CTHShortStorage"

renderHaskellType typeCat _ THIntStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHIntStorage)"
  FunctionParam -> Just $ "Ptr CTHIntStorage"

renderHaskellType typeCat _ THLongStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHLongStorage)"
  FunctionParam -> Just $ "Ptr CTHLongStorage"

renderHaskellType typeCat _ THHalfStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHHalfStorage)"
  FunctionParam -> Just $ "Ptr CTHHalfStorage"

renderHaskellType typeCat _ THCharStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHCharStorage)"
  FunctionParam -> Just $ "Ptr CTHCharStorage"

renderHaskellType typeCat _ THFloatStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHFloatStorage)"
  FunctionParam -> Just $ "Ptr CTHFloatStorage"

renderHaskellType typeCat _ THDoubleStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTHDoubleStorage)"
  FunctionParam -> Just $ "Ptr CTHDoubleStorage"

{- Other -}

renderHaskellType typeCat _ THGeneratorPtr = case typeCat of
  ReturnValue -> Just ("IO (Ptr CTHGenerator)") -- concrete type found in TensorMat)h
  FunctionParam -> Just ("Ptr CTHGenerator") -- concrete type found in TensorMath

renderHaskellType typeCat _ THAllocatorPtr = case typeCat of
  ReturnValue -> Just $ "IO (CTHAllocatorPtr)"
  FunctionParam -> Just $ "CTHAllocatorPtr"

renderHaskellType typeCat _ THDoublePtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CDouble)"
  FunctionParam -> Just "Ptr CDouble"

renderHaskellType _ _ THDouble =
  Just "CDouble" -- added from TensorRandom

renderHaskellType typeCat templateType THPtrDiff = case typeCat of
  ReturnValue -> Just $ "CPtrdiff"
  FunctionParam -> Just $ "CPtrdiff"
  -- TODO check if it's appropriate to splice here

renderHaskellType typeCat _ THLongPtrPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr (Ptr CLong))"
  FunctionParam -> Just "Ptr (Ptr CLong)"

renderHaskellType typeCat _ THLongPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CLong)"
  FunctionParam -> Just "Ptr CLong"

renderHaskellType typeCat _ THFloatPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CFloat)"
  FunctionParam -> Just "Ptr CFloat"

renderHaskellType _ _ THFloat =
  Just "CFloat"

renderHaskellType _ _ THLong =
  Just "CLong"

renderHaskellType typeCat _ THIntPtr = case typeCat of
  ReturnValue -> Just "IO (CIntPtr)"
  FunctionParam -> Just "CIntPtr"

renderHaskellType _ _ THInt =
  Just "CInt"

renderHaskellType _ _ THSize =
  Just "CSize"

renderHaskellType typeCat _ THCharPtrPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr (Ptr CChar))"
  FunctionParam -> Just "Ptr (Ptr CChar)"

renderHaskellType typeCat _ THCharPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CChar)"
  FunctionParam -> Just "Ptr CChar"

renderHaskellType _ _ THChar =
  Just "CChar"

renderHaskellType typeCat _ THShortPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CShort)"
  FunctionParam -> Just "Ptr CShort"

renderHaskellType _ _ THShort =
  Just "CShort"

renderHaskellType typeCat _ THHalfPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHHalf)"
  FunctionParam -> Just "Ptr CTHHalf"

renderHaskellType _ _ THHalf =
  Just "CTHHalf"

renderHaskellType typeCat templateType THRealPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr " <> realtype2Haskell templateType <> ")"
  FunctionParam -> Just $ "Ptr " <> realtype2Haskell templateType

renderHaskellType _ templateType THReal =
  Just $ realtype2Haskell templateType

renderHaskellType typeCat templateType THAccRealPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr " <> accrealtype2Haskell templateType <> ")"
  FunctionParam -> Just $ "Ptr " <> accrealtype2Haskell templateType

renderHaskellType _ templateType THAccReal =
  Just $ accrealtype2Haskell templateType

renderHaskellType typeCat _ THFilePtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHFile)"
  FunctionParam -> Just "Ptr CTHFile"

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> " #-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions =
  (T.intercalate "\n" (renderExtension <$> extensions)) <> "\n\n"

renderModule :: HModule -> Text
renderModule moduleSpec =
  "module " <> (renderModuleName moduleSpec)

renderExports :: [Text] -> Text
renderExports exports = (" (\n    "
                         <> (T.intercalate ",\n    " exports)
                         <> ") where\n\n")

renderImports :: [Text] -> Text
renderImports imports = (T.intercalate "\n" (singleimport <$> imports)) <> "\n\n"
  where singleimport x = "import " <> x

renderFunName :: Text -> Text -> Text
renderFunName prefix name = prefix <> "_" <> name


-- |Render a single function signature. Torch never calls back into haskell, so
-- unsafe is appropriate here
renderFunSig :: FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunSig headerFile modTypeTemplate (name, retType, args) =
  (
   "-- |c_" <> name <> " : "
   <> (T.intercalate " " nameSignature) <> " -> " <> (renderCType retType) <> "\n"
   <> "foreign import ccall unsafe \"" <> T.pack headerFile <> " " <> name <> "\"\n"
   <> "  c_" <> name <> " :: "
   <> (T.intercalate " -> " typeSignatureClean)
    -- TODO : fromJust shouldn't fail but still clean this up so it's not unsafe
   <> retArrow <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType)
  )
  where
    typeVals = thArgType <$> args
    typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
    typeSignatureClean = catMaybes typeSignature
    numArgs = P.length typeSignatureClean
    retArrow = if numArgs == 0 then "" else " -> "
    nameSignature = thArgName <$> args

-- |Render function pointer signature
renderFunPtrSig :: FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunPtrSig headerFile modTypeTemplate (name, retType, args) =
  (
   "-- |p_" <> name <> " : Pointer to "
   <> (T.intercalate " " nameSignature) <> " -> " <> (renderCType retType) <> "\n"
   <> "foreign import ccall unsafe \"" <> T.pack headerFile <> " &" <> name <> "\"\n"
   <> "  p_" <> name <> " :: FunPtr ("
   <> (T.intercalate " -> " typeSignatureClean)
    -- TODO : fromJust shouldn't fail but still clean this up so it's not unsafe
   <> retArrow <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType)
   <> ")"
  )
  where
    typeVals = thArgType <$> args
    typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
    typeSignatureClean = catMaybes typeSignature
    numArgs = P.length typeSignatureClean
    retArrow = if numArgs == 0 then "" else " -> "
    nameSignature = thArgName <$> args

-- TODO clean up redundancy of valid functions vs. functions in moduleSpec
renderFunctions :: HModule -> [THFunction] -> Text
renderFunctions moduleSpec@HModule{..} validFunctions =
  -- iteration over all functions
  intercalate "\n\n" (((renderFunSig modHeader typeTemplate)
                       <$> (P.zip3 funNames retTypes args))
                      <> ((renderFunPtrSig modHeader typeTemplate)
                          <$> (P.zip3 funNames retTypes args))
                     )
  where
    modulePrefix = modPrefix <> (type2SpliceReal modTypeTemplate) <> modSuffix <> "_"
    funNames = if modIsTemplate then
                 (mappend modulePrefix) <$> funName <$> validFunctions
               else
                 funName <$> validFunctions
    retTypes = funReturn <$> validFunctions
    args = funArgs <$> validFunctions
    typeTemplate = modTypeTemplate

-- |Check for conditional templating of functions and filter function list
checkList :: [THFunction] -> TemplateType -> [THFunction]
checkList fList templateType =
  P.filter ((checkFunction templateType) . funName) fList

renderAll :: HModule -> Text
renderAll spec@HModule{..} =
  (renderExtensions modExtensions
   <> renderModule spec
   <> renderExports exportFunctions
   <> renderImports modImports
   <> renderFunctions spec validFunctions)
  where
    prefix = makePrefix . type2SpliceReal $ modTypeTemplate
    bindings = modBindings
    splice = modPrefix <> (type2SpliceReal modTypeTemplate) <> modSuffix
    validFunctions = checkList modBindings modTypeTemplate
    exportFunctions =
      if modIsTemplate then
        ((renderFunName ("c_" <> splice) <$> (fmap funName (validFunctions)))
         <> (renderFunName ("p_" <> splice) <$> (fmap funName (validFunctions))))
      else
        ((renderFunName "c" <$> (fmap funName (validFunctions)))
         <> (renderFunName "p" <$> (fmap funName (validFunctions))))

renderCHeaderFile ::
  TemplateType -> [THFunction] -> (TemplateType -> [THFunction] -> HModule) -> IO ()
renderCHeaderFile templateType parsedBindings makeConfig = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile (outDir ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
  where modSpec = makeConfig templateType parsedBindings
        filename = (renderModuleName modSpec) <> ".hs"
        outDir = T.unpack (modOutDir modSpec)

renderModuleName :: HModule -> Text
renderModuleName HModule{..} =
  modPrefix <> (type2SpliceReal modTypeTemplate) <> modFileSuffix

-- ----------------------------------------
-- Execution
-- ----------------------------------------

-- |Remove If list was returned, extract non-Nothing values, o/w empty list
cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList (Left _) = []
cleanList (Right lst) = fromJust <$> (P.filter f lst)
  where
    f Nothing = False
    f (Just _) = True

parseFile :: [Char] -> IO [THFunction]
parseFile file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thParseGeneric file
  pure $ cleanList res
  where
    parseFromFile p file = runParser p file <$> readFile file
