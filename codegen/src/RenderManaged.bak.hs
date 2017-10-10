{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude as P

import Data.Text
import Data.List (nub)
import Text.Megaparsec
import Text.Show.Pretty

import Text.RawString.QQ

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared

outDirManaged = "./output/tensor/src/" :: Text

apply0String = [r|

-- | Generalized 1-argument managed operation
applyTX :: (Ptr CTHDoubleTensor -> a) -> TensorDouble_ -> a
applyTX operation tensor = unsafePerformIO $ do
  withForeignPtr (tdTensor tensor) (\t -> pure $ operation t)

|]

files :: [(String, TemplateType -> [THFunction] -> HModule)]
files =
  [
    ("vendor/torch7/lib/TH/generic/THTensor.h",
     (makeModule outDirManaged True "THTensor.h" "Tensor" "Tensor")),
    ("vendor/torch7/lib/TH/generic/THTensorMath.h",
     (makeModule outDirManaged True "THTensorMath.h" "Tensor" "TensorMath"))
  ]

data FunCategory =
  C_TR -- tensor -> real
  | C_TA -- tensor -> accreal
  | C_TV -- tensor -> void (in-place)
  | C_TRV -- tensor x real -> void (transform in-place, overwrite)
  | C_TTV -- tensor x tensor -> void (apply function to input vector)
  | C_TTRV -- tensor x tensor x real -> void
  | C_TRTRTT
  | C_Unknown -- no parse yet
  deriving (Eq, Show)

interpret (THFunction name args ret)
  | (P.length args == 1)
    && ((thArgType <$> args) == [THTensorPtr])
    && (ret == THReal) = C_TR
  | (P.length args == 1)
    && ((thArgType <$> args) == [THTensorPtr])
    && (ret == THAccReal) = C_TA
  | (P.length args == 1)
    && ((thArgType <$> args) == [THTensorPtr])
    && (ret == THVoid) = C_TV
  | (P.length args == 2)
    && ((thArgType <$> args) == [THTensorPtr, THReal])
    && (ret == THVoid) = C_TRV
  | (P.length args == 2)
    && ((thArgType <$> args) == [THTensorPtr, THTensorPtr])
    && (ret == THVoid) = C_TTV
  | (P.length args == 3)
    && ((thArgType <$> args) == [THTensorPtr, THTensorPtr, THReal])
    && (ret == THVoid) = C_TTRV
  | otherwise = C_Unknown

-- runPipelineManaged ::
--   [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipelineManaged headerPath makeModuleConfig typeList = do
  let headerPath = "vendor/torch7/lib/TH/generic/THTensorMath.h"
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  let interpretation = interpret <$> bindingsUniq
  mapM_ print interpretation
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  pure ()
--   mapM_ (\x -> renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
--   putStrLn $ "Number of functions generated: " ++
--     (show $ P.length typeList * P.length bindingsUniq)

main = do
  mapM_ (\(file, spec) -> runPipelineManaged file spec files) files
  putStrLn "Done"
