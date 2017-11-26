{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ProcessSpec where

import Data.Text
import Data.Yaml
import GHC.Generics
import Data.Aeson.Encode.Pretty
-- import qualified Data.Lazy.ByteString as B

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

-- yaml ATen spec
-- specFile = "vendor/aten/src/ATen/Declarations.cwrap"
specFile = "vendor/declarations.yaml"
-- specFile = "vendor/aten/src/ATen/dummy.cwrap"

main = do
  (result :: Either ParseException Value) <- decodeFileEither specFile
  case result of
    (Left e) -> putStrLn $ prettyPrintParseException e
    (Right x) -> putStrLn $ show x -- putStrLn $ show $ encodePretty x
  putStrLn "Done"

