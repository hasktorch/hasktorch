{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Control.Monad (void)
import Data.List (nub)
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

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared

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
  ReturnValue -> Just ("IO (Ptr CTHGenerator") -- concrete type found in TensorMat)h
  FunctionParam -> Just ("Ptr CTHGenerator") -- concrete type found in TensorMath

renderHaskellType typeCat _ THAllocatorPtr = case typeCat of
  ReturnValue -> Just $ "IO (CTHAllocatorPtr)"
  FunctionParam -> Just $ "CTHAllocatorPtr"

renderHaskellType _ _ THDouble =
  Just "CDouble" -- added from TensorRandom

renderHaskellType typeCat templateType THPtrDiff = case typeCat of
  ReturnValue -> Just $ "IO (CTH" <> type2SpliceReal templateType <> "PtrDiff)"
  FunctionParam -> Just $ "CTH" <> type2SpliceReal templateType <> "PtrDiff"
  -- TODO check if it's appropriate to splice here

renderHaskellType typeCat _ THLongPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CLong)"
  FunctionParam -> Just "Ptr CLong"

renderHaskellType _ _ THLong =
  Just "CLong"

renderHaskellType typeCat _ THIntPtr = case typeCat of
  ReturnValue -> Just "IO (CIntPtr)"
  FunctionParam -> Just "CIntPtr"

renderHaskellType _ _ THInt =
  Just "CInt"

renderHaskellType _ _ THSize =
  Just "CSize"

renderHaskellType typeCat templateType THCharPtr = case typeCat of
  ReturnValue -> Just "IO (Ptr CChar)"
  FunctionParam -> Just "Ptr CChar"

renderHaskellType _ _ THChar =
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
  (T.intercalate "\n" (renderExtension <$> extensions)) <> "\n\n"

renderModuleName :: HModule -> Text
renderModuleName HModule{..} =
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

-- TODO clean up redundancy of valid functions vs. functions in moduleSpec
renderFunctions :: HModule -> [THFunction] -> Text
renderFunctions moduleSpec@HModule{..} validFunctions =
  -- iteration over all functions
  intercalate "\n\n" ((renderFunSig modHeader typeTemplate)
                      <$> (P.zip3 funNames retTypes args) ) 
  where
    -- modulePrefix = (renderModuleName moduleSpec) <> "_"
    modulePrefix = modPrefix <> (type2SpliceReal modTypeTemplate) <> modSuffix <> "_"
    -- funNames = (mappend modulePrefix) <$> funName <$> modBindings
    funNames = (mappend modulePrefix) <$> funName <$> validFunctions
    retTypes = funReturn <$> modBindings
    args = funArgs <$> modBindings
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
--       (renderFunName ("c_" <> renderModuleName spec)
      (renderFunName ("c_" <> splice)
       <$> (fmap funName (validFunctions)))

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

renderCHeader templateType parsedBindings makeConfig = do
  putStrLn $ "Writing " <> T.unpack filename
  writeFile ("./output/" ++ T.unpack filename) (T.unpack . renderAll $ modSpec)
  where modSpec = makeConfig templateType parsedBindings
        filename = (renderModuleName modSpec) <> ".hs"

runPipeline ::
  [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipeline headerPath makeModuleConfig typeList = do
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  -- TODO nub is a hack until proper treatment of conditioned templates is implemented
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  mapM_ (\x -> renderCHeader x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length typeList * P.length bindingsUniq)

-- TODO re-factor to unify w/ parseFile
parseFileConcrete :: [Char] -> IO [THFunction]
parseFileConcrete file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thParseConcrete file
  pure $ cleanList res
  where
    parseFromFile p file = runParser p file <$> readFile file

-- TODO re-factor to unify w/ runPipeline
runPipelineConcrete ::
  [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipelineConcrete headerPath makeModuleConfig typeList = do
  parsedBindings <- parseFileConcrete headerPath
  let bindingsUniq = nub parsedBindings
  -- TODO nub is a hack until proper treatment of conditioned templates is implemented
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  mapM_ (\x -> renderCHeader x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length typeList * P.length bindingsUniq)

genericFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
genericFiles =
  [
    ("vendor/torch7/lib/TH/generic/THBlas.h",
     (makeModule "THBlas.h" "Blas" "Blas")),
    ("vendor/torch7/lib/TH/generic/THLapack.h",
     (makeModule "THLapack.h" "Lapack" "Lapack")),
    ("vendor/torch7/lib/TH/generic/THStorage.h",
     (makeModule "THStorage.h" "Storage" "Storage")),
    ("vendor/torch7/lib/TH/generic/THStorageCopy.h",
     (makeModule "THStorageCopy.h" "Storage" "StorageCopy")),
    ("vendor/torch7/lib/TH/generic/THTensor.h",
     (makeModule "THTensor.h" "Tensor" "Tensor")),
    ("vendor/torch7/lib/TH/generic/THTensorConv.h",
     (makeModule "THTensorConv.h" "Tensor" "TensorConv")),
    ("vendor/torch7/lib/TH/generic/THTensorCopy.h",
     (makeModule "THTensorCopy.h" "Tensor" "TensorCopy")),
    ("vendor/torch7/lib/TH/generic/THTensorLapack.h",
     (makeModule "THTensorLapack.h" "Tensor" "TensorLapack")),
    ("vendor/torch7/lib/TH/generic/THTensorMath.h",
     (makeModule "THTensorMath.h" "Tensor" "TensorMath")),
    ("vendor/torch7/lib/TH/generic/THTensorRandom.h",
     (makeModule "THTensorRandom.h" "Tensor" "TensorRandom")),
    ("vendor/torch7/lib/TH/generic/THVector.h",
     (makeModule "THVector.h" "Vector" "Vector"))
  ]

concreteFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
concreteFiles =
  [
    ("vendor/check.h",
     (makeModule "THFile.h" "File" "File")),
    ("vendor/torch7/lib/TH/THFile.h",
     (makeModule "THFile.h" "File" "File")),
    ("vendor/torch7/lib/TH/THDiskFile.h",
     (makeModule "THDiskFile.h" "DiskFile" "DiskFile"))
  ]

-- |TODO: make a unified module that re-exports all functions
makeReExports = do
  putStrLn "Re-exported Tensors"

testString inp parser = case (parse parser "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

check :: IO ()
check = do
  testString exampleText thParseGeneric
  where
    exampleText = "skip this garbage line line\n" <>
      "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
      "another garbage line ( )@#R @# 324 32"

check2 :: IO ()
check2 = do
  testString exampleText thParseGeneric
  where
    exampleText =
      "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);"

check3 :: IO ()
check3 = do
  testString exampleText thParseConcrete
  where
    exampleText =
      "TH_API void THTensor_fooasdf(THTensor *self,const char flag);"

main :: IO ()
main = do
  mapM_ (\(file, spec) -> runPipeline file spec genericTypes) genericFiles
  -- mapM_ (\(file, spec) -> runPipelineConcrete file spec concreteTypes) concreteFiles
  putStrLn "Done"
