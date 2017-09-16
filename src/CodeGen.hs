{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (void)
import Data.Monoid ((<>))
import Data.Maybe
import Data.Void
import Data.Text
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Prelude as P
import Text.Show.Pretty

import CodeParse

-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data TemplateType = GenByte
                  | GenChar
                  | GenDouble
                  | GenFloat
                  | GenHalf
                  | GenInt
                  | GenLong
                  | GenShort deriving Show

-- List used to iterate through all template types
genTypes = [GenByte, GenChar,
            GenDouble, GenFloat, GenHalf,
            GenInt, GenLong, GenShort] :: [TemplateType]

data HModule = HModule {
  modHeader :: FilePath,
  modPrefix :: Text,
  modTypeTemplate :: TemplateType,
  modSuffix :: Text,
  modFileSuffix :: Text,
  modExtensions :: [Text],
  modImports :: [Text],
  modTypeDefs :: [(Text, Text)],
  modBindings :: [THFunction]
  } deriving Show

data TypeCategory = ReturnValue | FunctionParam

-- ----------------------------------------
-- Rendering
-- ----------------------------------------

makePrefix :: Text -> Text
makePrefix templateType = "TH" <> templateType <> "Tensor"

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

-- #define real [X]
type2real :: TemplateType -> Text
type2real GenByte   = "unsigned char"
type2real GenChar   = "char"
type2real GenDouble = "double"
type2real GenFloat  = "float"
type2real GenHalf   = "THHalf"
type2real GenInt    = "int"
type2real GenLong   = "long"
type2real GenShort  = "short"

-- #define accreal [X]
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

renderCType :: THType -> Text
renderCType THVoid = "void"
renderCType THDescBuff = "THDescBuff"
renderCType THTensorPtr = "THTensor *"
renderCType THTensorPtrPtr = "THTensor **"
renderCType THByteTensorPtr = "THByteTensor *"
renderCType THLongTensorPtr = "THLongTensor *"
renderCType THDoubleTensorPtr = "THDoubleTensor *"
renderCType THFloatTensorPtr = "THFloatTensor *"
renderCType THGeneratorPtr = "THGenerator *"
renderCType THStoragePtr = "THStorage *"
renderCType THLongStoragePtr = "THLongStorage *"
renderCType THPtrDiff = "ptrdiff_t"
renderCType THLongPtr = "long *"
renderCType THLong = "long"
renderCType THIntPtr = "int *"
renderCType THInt = "int"
renderCType THSize = "size_t"
renderCType THCharPtr = "char *"
renderCType THChar = "char"
renderCType THRealPtr = "real *"
renderCType THReal = "real"
renderCType THAccRealPtr = "accreal *"
renderCType THAccReal = "accreal"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text

renderHaskellType _ templateType THVoidPtr = Just "Ptr ()"

renderHaskellType typeCat templateType THVoid =
  case typeCat of
    ReturnValue -> Just "IO ()"
    FunctionParam -> Nothing

renderHaskellType _ _ THDescBuff = Just "CTHDescBuff"

renderHaskellType typeCat templateType THTensorPtrPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor))"
  FunctionParam -> Just $ "Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType typeCat templateType THTensorPtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"
  FunctionParam -> Just $ "(Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType typeCat templateType THByteTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHByteTensor)"
  FunctionParam -> Just "Ptr CTHByteTensor"

renderHaskellType typeCat templateType THLongTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHLongTensor)"
  FunctionParam -> Just "Ptr CTHLongTensor"

renderHaskellType typeCat templateType THDoubleTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHDoubleTensor)"
  FunctionParam -> Just "Ptr CTHDoubleTensor"

renderHaskellType typeCat templateType THFloatTensorPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CTHFloatTensor)"
  FunctionParam -> Just "Ptr CTHFloatTensor"

renderHaskellType typeCat templateType THGeneratorPtr = case typeCat of
  ReturnValue -> Just ("IO (Ptr CTHGenerator") -- concrete type found in TensorMat)h
  FunctionParam -> Just ("Ptr CTHGenerator") -- concrete type found in TensorMath

renderHaskellType typeCat templateType THStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTH" <> type2SpliceReal templateType <> "Storage)"
  FunctionParam -> Just $ "Ptr CTH" <> type2SpliceReal templateType <> "Storage"

renderHaskellType typeCat templateType THLongStoragePtr = case typeCat of
  ReturnValue -> Just $ "IO (Ptr CTH" <> type2SpliceReal templateType <> "LongStorage)"
  FunctionParam -> Just $ "Ptr CTH" <> type2SpliceReal templateType <> "LongStorage"

renderHaskellType typeCat templateType THAllocatorPtr = case typeCat of
  ReturnValue -> Just $ "IO (CTHAllocatorPtr)"
  FunctionParam -> Just $ "CTHAllocatorPtr"

renderHaskellType _ templateType THDouble =
  Just "CDouble" -- added from TensorRandom

renderHaskellType typeCat templateType THPtrDiff = case typeCat of
  ReturnValue -> Just $ "IO (CTH" <> type2SpliceReal templateType <> "PtrDiff)"
  FunctionParam -> Just $ "CTH" <> type2SpliceReal templateType <> "PtrDiff"
  -- TODO check if it's appropriate to splice here

renderHaskellType typeCat templateType THLongPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CLong)"
  FunctionParam -> Just "Ptr CLong"

renderHaskellType _ templateType THLong =
  Just "CLong"

renderHaskellType typeCat templateType THIntPtr = case typeCat of
  ReturnValue -> Just "IO (CIntPtr)"
  FunctionParam -> Just "CIntPtr"

renderHaskellType _ templateType THInt =
  Just "CInt"

renderHaskellType _ templateType THSize =
  Just "CSize"

renderHaskellType typeCat templateType THCharPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CChar)"
  FunctionParam -> Just "Ptr CChar"

renderHaskellType _ templateType THChar =
  Just "CChar"

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

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> "#-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions =
  (T.intercalate "\n" (renderExtension <$> extensions)) <> "\n"

renderModuleName :: HModule -> Text
renderModuleName HModule{..} =
  modPrefix <> (type2SpliceReal modTypeTemplate) <> modFileSuffix

-- renderModuleFilename :: HModule -> Text
-- renderModuleFilename HModule{..} =
--   modPrefix <> (type2SpliceReal modTypeTemplate) <> modFileSuffix

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

renderFunSig :: FilePath -> TemplateType -> (Text, THType, [THArg]) -> Text
renderFunSig headerFile modTypeTemplate (name, retType, args) =
  (
   "-- |c_" <> name <> " : "
   <> (T.intercalate " " nameSignature) <> " -> " <> (renderCType retType) <> "\n"
   <> "foreign import ccall \"" <> T.pack headerFile <> " " <> name <> "\"\n"
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

renderFunctions :: HModule -> Text
renderFunctions moduleSpec@HModule{..} =
  -- iteration over all functions
  intercalate "\n\n" ((renderFunSig modHeader typeTemplate)
                      <$> (P.zip3 funNames retTypes args) )
  where
    modulePrefix = (renderModuleName moduleSpec) <> "_"
    funNames = (mappend modulePrefix) <$> funName <$> modBindings
    retTypes = funReturn <$> modBindings
    args = funArgs <$> modBindings
    typeTemplate = modTypeTemplate

renderAll :: HModule -> Text
renderAll spec =
  (renderExtensions (modExtensions spec)
   <> renderModule spec
   <> renderExports exportFunctions
   <> renderImports (modImports spec)
   <> renderFunctions spec)
  where
    prefix = makePrefix . type2SpliceReal . modTypeTemplate $ spec
    bindings = modBindings spec
    exportFunctions =
      (renderFunName ("c_" <> renderModuleName spec)
       <$> (fmap funName (modBindings spec)))

-- ----------------------------------------
-- Execution
-- ----------------------------------------

parseFromFile p file = runParser p file <$> readFile file

cleanList :: Either (ParseError Char Void) [Maybe THFunction] -> [THFunction]
cleanList (Left _) = []
cleanList (Right lst) = fromJust <$> (P.filter f lst)
  where
    f Nothing = False
    f (Just _) = True

makeModule modHeader modSuffix modFileSuffix typeTemplate bindings =
   HModule {
        modHeader = modHeader,
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = modSuffix,
        modFileSuffix = modFileSuffix,
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }

parseFile file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thFile file
  pure $ cleanList res

renderCHeader templateType parsedBindings makeConfig = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile ("./th-bindings/" ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
  where modSpec = makeConfig templateType parsedBindings
        filename = (renderModuleName modSpec) <> ".hs"

runPipeline headerPath makeModuleConfig = do
  parsedBindings <- parseFile headerPath
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 parsedBindings)
  mapM_ (\x -> renderCHeader x parsedBindings makeModuleConfig) genTypes
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length genTypes * P.length parsedBindings)

parseFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
parseFiles =
  [
    ("vendor/torch7/lib/TH/generic/THBlas.h",
     (makeModule "THBlas.h" "Blas" "Blas")),
    ("vendor/torch7/lib/TH/generic/THLapack.h",
     (makeModule "THLapack.h" "Lapack" "Lapack")),
    ("vendor/torch7/lib/TH/generic/THStorage.h",
     (makeModule "THStorage.h" "Storage" "Storage")),
    ("vendor/torch7/lib/TH/generic/THTensor.h",
     (makeModule "THTensor.h" "Tensor" "Tensor")),
    ("vendor/torch7/lib/TH/generic/THTensorConv.h",
     (makeModule "THTensorConv.h" "Tensor" "TensorConv")),
    ("vendor/torch7/lib/TH/generic/THTensorMath.h",
     (makeModule "THTensorMath.h" "Tensor" "TensorMath")),
    ("vendor/torch7/lib/TH/generic/THTensorRandom.h",
     (makeModule "THTensorRandom.h" "Tensor" "TensorRandom")),
    ("vendor/torch7/lib/TH/generic/THVector.h",
     (makeModule "THVector.h" "Vector" "Vector"))
  ]

testString inp = case (parse thFile "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

test1 = do
  testString ex1
  where
    ex1 = "skip this garbage line line\n" <>
     "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
     "another garbage line ( )@#R @# 324 32"


test2 = runPipeline "vendor/check.h"
  (makeModule "THStorage.h" "Storage" "Storage")

-- |TODO unfinished/nonfunctional parses
todo = do
  runPipeline "vendor/torch7/lib/TH/generic/THStorageCopy.h"
    (makeModule "THStorageCopy.h" "Storage" "StorageCopy")
  runPipeline "vendor/torch7/lib/TH/generic/THTensorCopy.h"
    (makeModule "THTensorCopy.h" "Tensor" "TensorCopy")
  runPipeline "vendor/torch7/lib/TH/generic/THTensorLapack.h"
    (makeModule "THTensorLapack.h" "Tensor" "TensorLapack")

main = do
  mapM_ (\(file, spec) -> runPipeline file spec) parseFiles
  putStrLn "Done"
