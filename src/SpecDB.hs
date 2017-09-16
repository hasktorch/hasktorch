{-# LANGUAGE OverloadedStrings #-}

module Main where

import           Control.Applicative
import qualified Data.Text as T
import           Database.SQLite.Simple
import           Database.SQLite.Simple.FromRow

import Prelude hiding (catch)
import System.Directory
import Control.Exception
import System.IO.Error hiding (catch)

import CodeGenTypes as Y

data TestField = TestField Int T.Text deriving (Show)

instance FromRow TestField where
  fromRow = TestField <$> field <*> field

instance ToRow TestField where
  toRow (TestField id_ str) = toRow (id_, str)

removeIfExists :: FilePath -> IO ()
removeIfExists fileName = removeFile fileName `catch` handleExists
  where handleExists e
          | isDoesNotExistError e = return ()
          | otherwise = throwIO e

makeTypeTable conn = do
  mapM_ go (type2SpliceReal <$> Y.genTypes)
  where
    cmd = "INSERT INTO template_types (types) VALUES (?)"
    go typename =
      execute conn cmd (Only (T.unpack typename :: String))

main :: IO ()
main = do
  removeIfExists "specdb/specdb.db"
  conn <- open "specdb/specdb.db"
  execute_ conn "CREATE TABLE IF NOT EXISTS tests (id INTEGER PRIMARY KEY, test TEXT)"
  execute_ conn "CREATE TABLE IF NOT EXISTS template_types (id INTEGER PRIMARY KEY, types TEXT)"
  makeTypeTable conn
  execute conn "INSERT INTO tests (test) VALUES (?)" (Only ("this is some test" :: String))
  putStrLn "tests table:"
  r <- query_ conn "SELECT * from tests" :: IO [TestField]
  mapM_ print r
  putStrLn "types table:"
  r <- query_ conn "SELECT * from template_types" :: IO [TestField]
  mapM_ print r
  close conn
