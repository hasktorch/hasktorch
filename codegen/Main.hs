{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Options.Applicative as O
import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Text.Megaparsec as M
import Text.Show.Prettyprint (prettyPrint)

{- native_functions.yaml -}

data NativeFunction = NativeFunction {
  func :: String
  , variants :: Maybe String
  , python_module :: Maybe String
  , device_guard :: Maybe Bool
  , dispatch :: Maybe Dispatch
  -- , dispatch :: Maybe DispatchSum
  , requires_tensor :: Maybe Bool
} deriving (Show, Generic)

-- data DispatchSum = DStruct Dispatch | SStruct String deriving (Show, Generic, FromJSON, ToJSON)
-- data DispatchSum = DStruct Dispatch | SStruct String deriving (Show, Generic)
-- data DispatchSum = Either Dispatch String deriving (Show, Generic)
-- instance FromJSON DispatchSum

-- data Dispatch = Dispatch {
--   cpu :: Maybe String
--   , gpu :: Maybe String
--   , cuda :: Maybe String
--   , sparseCPU :: Maybe String
--   , sparseCUDA :: Maybe String
-- } | String deriving (Show, Generic)


data Dispatch = Dispatch {
  cpu :: Maybe String
  , gpu :: Maybe String
  , cuda :: Maybe String
  , sparseCPU :: Maybe String
  , sparseCUDA :: Maybe String
} deriving (Show, Generic)

dispatchModifier fieldName
  | fieldName `elem` ["cpu", "gpu", "cuda"] = upper fieldName
  | fieldName == "sparseCPU" = "SparseCPU"
  | fieldName == "sparseGPU" = "SparseGPU"
  | fieldName == "sparseCUDA" = "SparseCUDA"
  | otherwise = fieldName
  where upper = map toUpper

instance FromJSON NativeFunction 
instance ToJSON NativeFunction

instance FromJSON Dispatch where
  parseJSON = genericParseJSON $
    defaultOptions {
      fieldLabelModifier = dispatchModifier
    }

instance ToJSON Dispatch


{- derivatives.yaml -}

data Derivative = Derivative {
  name :: String
  , grad_output :: Maybe String
  , output_differentiability :: [Bool]
  , self :: Maybe String
  , tensors :: Maybe String

} deriving (Show, Generic)

instance FromJSON Derivative

{- CLI options -}

data Options = Options
    { specFile :: !String
    } deriving Show

optsParser :: O.ParserInfo Options
optsParser = O.info
  (O.helper <*> versionOption <*> programOptions)
  ( O.fullDesc <> O.progDesc "ffi codegen" <> O.header
    "codegen for hasktorch 0.0.2"
  )

versionOption :: O.Parser (a -> a)
versionOption =
  O.infoOption "0.0.2" (O.long "version" <> O.help "Show version")

programOptions :: O.Parser Options
programOptions = Options <$> O.strOption
  (  O.long "spec-file"
  <> O.metavar "FILENAME"
  <> O.value "spec/small_test.yaml"
  <> O.help "Specification file"
  )

{- Execution -}


decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  prettyPrint file


-- main2 :: IO ()
-- main2 = do
--     print . midentity $ testdata d0
--     print . midentity $ testdata d1
--     putStrLn "success!"
--   where
--     midentity :: NativeFunction -> Either String NativeFunction
--     midentity x = (decodeEither "Fail") . (encode)
  
-- testdata :: DispatchSum -> NativeFunction
-- testdata ds = NativeFunction
--       { func            = "func"
--       , variants        = Nothing
--       , python_module   = Nothing
--       , device_guard    = Nothing
--       , dispatch        = Just ds
--       , requires_tensor = Nothing
--       }
  
-- d0 :: DispatchSum
-- d0 = SStruct "astring"
  
-- d1 :: DispatchSum
-- d1 = DStruct $ Dispatch Nothing Nothing Nothing Nothing Nothing

testdata :: Dispatch -> NativeFunction
testdata ds = NativeFunction
      { func            = "func"
      , variants        = Nothing
      , python_module   = Nothing
      , device_guard    = Nothing
      , dispatch        = Just ds
      , requires_tensor = Nothing
      }
  
-- d0 :: Dispatch
-- d0 = "astring"
  
d1 :: Dispatch
d1 = Dispatch Nothing Nothing Nothing Nothing Nothing


main :: IO ()
main = do
  opts <- O.execParser optsParser
  -- decodeAndPrint (specFile opts)
  -- decodeAndPrint "spec/native_functions.yaml"
  decodeAndPrint "spec/native_functions_modified.yaml"
  -- decodeAndPrint "spec/tiny.yaml"
  putStrLn "Done"
