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
renderCType THInt = "int"
renderCType THChar = "char"
renderCType THRealPtr = "real *"
renderCType THReal = "real"
renderCType THAccRealPtr = "accreal *"
renderCType THAccReal = "accreal"

renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType typeCat templateType THVoid =
  case typeCat of
    ReturnValue -> Just "IO ()"
    FunctionParam -> Nothing

renderHaskellType _ _ THDescBuff = Just "CTHDescBuff"

renderHaskellType _ templateType THTensorPtrPtr =
  Just $ "Ptr (Ptr CTH" <> type2SpliceReal templateType <> "Tensor)"

renderHaskellType _ templateType THTensorPtr =
  Just ("Ptr CTH" <> type2SpliceReal templateType)

renderHaskellType _ templateType THByteTensorPtr =
  Just ("Ptr CTHByteTensor") -- concrete type found in TensorMath

renderHaskellType _ templateType THLongTensorPtr =
  Just ("Ptr CTHLongTensor") -- concrete type found in TensorMath

renderHaskellType _ templateType THDoubleTensorPtr =
  Just ("Ptr CTHDoubleTensor") -- concrete type found in TensorRandom

renderHaskellType _ templateType THFloatTensorPtr =
  Just ("Ptr CTHFloatTensor") -- concrete type found in TensorRandom

renderHaskellType _ templateType THGeneratorPtr =
  Just ("Ptr CTHGenerator") -- concrete type found in TensorMath

renderHaskellType _ templateType THStoragePtr =
  Just $ "Ptr CTH" <> type2SpliceReal templateType <> "Storage"

renderHaskellType _ templateType THLongStoragePtr =
  Just $ "Ptr CTH" <> type2SpliceReal templateType <> "LongStorage"

renderHaskellType _ templateType THDouble =
  Just "CDouble" -- added from TensorRandom

renderHaskellType _ templateType THPtrDiff =
  Just $ "CTH" <> type2SpliceReal templateType <> "PtrDiff"
  -- TODO check if it's appropriate to splice here

renderHaskellType _ templateType THLongPtr =
  Just "Ptr CLong"

renderHaskellType _ templateType THLong =
  Just "CLong"

renderHaskellType _ templateType THInt =
  Just "CInt"

renderHaskellType _ templateType THChar =
  Just "CChar"

renderHaskellType _ templateType THRealPtr =
  Just $ "Ptr " <> realtype2Haskell templateType

renderHaskellType _ templateType THReal =
  Just $ realtype2Haskell templateType

renderHaskellType _ templateType THAccRealPtr =
  Just $ "Ptr " <> accrealtype2Haskell templateType

renderHaskellType _ templateType THAccReal =
  Just $ accrealtype2Haskell templateType

renderExtension :: Text -> Text
renderExtension extension = "{-# LANGUAGE " <> extension <> "#-}"

renderExtensions :: [Text] -> Text
renderExtensions extensions = T.intercalate "\n" (renderExtension <$> extensions)

renderModuleName :: HModule -> Text
renderModuleName HModule{..} =
  modPrefix <> (type2SpliceReal modTypeTemplate) <> modSuffix

renderModuleFilename :: HModule -> Text
renderModuleFilename HModule{..} =
  modPrefix <> (type2SpliceReal modTypeTemplate) <> modFileSuffix

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
   <> (T.intercalate " -> " $ catMaybes typeSignature)
   -- TODO : fromJust shouldn't fail, clean this up so it's not unsafe
   <> " -> " <> fromJust (renderHaskellType ReturnValue modTypeTemplate retType) <> "\n"
  )
  where
    typeVals = thArgType <$> args
    typeSignature = renderHaskellType FunctionParam modTypeTemplate <$> typeVals
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
    renderModule spec
    <> renderExports exportFunctions
    <> renderImports (modImports spec)
    <> renderFunctions spec
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

makeTensorModule typeTemplate bindings =
   HModule {
        modHeader = "THTensor.h",
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = "Tensor",
        modFileSuffix = "TensorMath",
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }

makeTensorMathModule typeTemplate bindings =
   HModule {
        modHeader = "THTensorMath.h",
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = "Tensor",
        modFileSuffix = "TensorMath",
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }

makeTensorRandomModule typeTemplate bindings =
   HModule {
        modHeader = "THTensorRandom.h",
        modPrefix = "TH",
        modTypeTemplate = typeTemplate,
        modSuffix = "Tensor",
        modFileSuffix = "TensorRandom",
        modExtensions = ["ForeignFunctionInterface"],
        modImports = ["Foreign", "Foreign.C.Types", "THTypes"],
        modTypeDefs = [],
        modBindings = bindings
  }

parseFile file = do
  putStrLn $ "\nRunning " ++ file ++ " ... "
  putStrLn $ "Parsing ... "
  res <- parseFromFile thFile file
  pure $ cleanList res

renderCHeader templateType parsedBindings makeConfig = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile ("./th-bindings/" ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
  where modSpec = makeConfig templateType parsedBindings
        filename = (renderModuleFilename modSpec) <> ".hs"

runPipeline headerPath makeModuleConfig = do
  parsedBindings <- parseFile headerPath
  putStrLn $ ppShow (P.take 3 parsedBindings)
  mapM_ (\x -> renderCHeader x parsedBindings makeModuleConfig) genTypes
  putStrLn $ "Number of functios: " ++ (show $ P.length genTypes
                                        * P.length parsedBindings)
  putStrLn "First 3 signatures"

testString inp = case (parse thFile "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

test1 = do
  testString ex1
  where
    ex1 = "skip this garbage line line\n" <>
     "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
     "another garbage line ( )@#R @# 324 32"

main = do

  runPipeline "vendor/torch7/lib/TH/generic/THTensor.h"
    makeTensorModule
  runPipeline "vendor/torch7/lib/TH/generic/THTensorMath.h"
    makeTensorMathModule
  runPipeline "vendor/torch7/lib/TH/generic/THTensorRandom.h"
    makeTensorRandomModule
  -- runPipeline "vendor/check.h" makeTensorModule

  -- TODO - other headers

  putStrLn "Done"
