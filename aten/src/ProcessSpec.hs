{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ProcessSpec where

import Data.Text hiding (take)
import Data.Yaml
import GHC.Generics
import Data.Aeson.Encode.Pretty
import qualified Data.ByteString.Lazy.Char8 as B

data Entry = Entry {
  name :: Text,
  python_name :: Text,
  cpu_half :: Bool,
  auto_gpu :: Bool,
  return :: Text,
  arguments :: [Text]
  } deriving (Eq, Generic, Show)

instance FromJSON Entry
instance ToJSON Entry

-- -- original file - not a properyaml file:
-- specFile = "vendor/aten/src/ATen/Declarations.cwrap"

-- yaml-ified ATen spec:
specFile = "vendor/aten-declarations.yaml"

main = do
  (result :: Either ParseException Value) <- decodeFileEither specFile
  case result of
    (Left e) -> putStrLn $ prettyPrintParseException e
    (Right x) -> putStrLn . (take 600) . B.unpack . encodePretty $ x
  putStrLn ".\n.\n.\n"
  putStrLn "Done"

