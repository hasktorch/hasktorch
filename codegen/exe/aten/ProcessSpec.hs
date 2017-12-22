{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.Either (fromRight)
import Data.Maybe
import Data.Text (Text)
import Data.Yaml
import GHC.Generics
import Data.Aeson.Encode.Pretty
import Data.Aeson.Types
import qualified Data.ByteString.Lazy.Char8 as B

-- use for `backends` and `backend_type_pairs` fields
data Backend = CPU | Float | Double | CUDA deriving (Eq, Show)

-- values of `variants` fields
data Variants = Function | Method deriving (Eq, Show)

-- values of `return` fields
data Return = Long | Argument Int | THTensorPtr | Self deriving (Eq, Show)

data Entry = Entry {
  -- Common fields
  aten_name :: Text,
  aten_return :: Maybe Text,
  aten_arguments :: Maybe [Value],
  aten_backends :: Maybe [Text],
  aten_backend_type_pairs :: Maybe [[Text]],
  -- Secondary fields
  aten_cname :: Maybe Text,
  aten_options :: Maybe [Object],
  aten_with_gil :: Maybe Bool,
  aten_cpu_half :: Maybe Bool,
  aten_auto_gpu :: Maybe Bool,

  aten_before_call :: Maybe Text,
  aten_variants :: Maybe [Text],
  -- python-specific ops
  aten_scalar_check :: Maybe Text,
  aten_python_name :: Maybe Text,
  aten_aten_custom_call :: Maybe Text,
  aten_before_arg_assign :: Maybe Text
  } deriving (Eq, Generic, Show)

customOptions = defaultOptions 
  {
    -- yaml fromat drops "aten_" prefix
    fieldLabelModifier = Prelude.drop 5
  }

instance FromJSON Entry where
    parseJSON = genericParseJSON customOptions

instance ToJSON Entry where
    toJSON     = genericToJSON customOptions
    toEncoding = genericToEncoding customOptions

-- yaml-ified ATen spec, generated from vendor/aten/src/ATen/Declarations.cwrap
-- using the script vendor/build-aten-declarations.sh
specFile = "vendor/aten-spec/Declarations.yaml"

-- |given an optional accessor filter entries for entries where that accessor is
-- not nothing
filt :: forall a . (Entry -> Maybe a) -> [Entry] -> [Entry]
filt f dat = filter
  (\x -> case (f x) of
      Just y -> True
      Nothing -> False
  ) dat

pp :: ToJSON a => a -> IO ()
pp x = putStrLn . B.unpack . encodePretty $ x

readYaml :: IO (Either ParseException [Entry])
readYaml = decodeFileEither specFile

getDat :: IO [Entry]
getDat = fromRight undefined <$> readYaml

-- |extract functions with cname
cFunctions :: [Entry] -> [Entry]
cFunctions dat = filt aten_cname dat

explore :: IO ()
explore = do
  dat <- getDat
  pp $ head dat
  print $ length $ cFunctions dat
  pp $ take 4 $ cFunctions dat
  mapM_ (\x -> putStrLn $ show . fromJust $ x) (aten_cname <$> (cFunctions dat))
  pure ()

main :: IO ()
main = do
  (result :: Either ParseException [Entry]) <- decodeFileEither specFile
  case result of
    (Left e) -> putStrLn $ prettyPrintParseException e
    (Right x) -> putStrLn . (take 1000) . B.unpack . encodePretty $ x
  putStrLn ".\n.\n.\n"
  putStrLn "Done"
