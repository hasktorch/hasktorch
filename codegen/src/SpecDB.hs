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

import CodeGenTypes (genericTypes)
import RenderShared (type2SpliceReal)

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

createDB = do
  removeIfExists "output/specdb.db"
  conn <- open "output/specdb.db"
  pure conn

createTables conn = do
  execute_ conn "CREATE TABLE IF NOT EXISTS tests (id INTEGER PRIMARY KEY, test TEXT)"
  execute_ conn "CREATE TABLE IF NOT EXISTS template_types (id INTEGER PRIMARY KEY, types TEXT)"

makeTypeTable conn = do
  mapM_ go (type2SpliceReal <$> genericTypes)
  where
    cmd = "INSERT INTO template_types (types) VALUES (?)"
    go typename =
      execute conn cmd (Only (T.unpack typename :: String))

makeTests conn = do
  execute conn "INSERT INTO tests (test) VALUES (?)" (Only ("this is some test" :: String))

main :: IO ()
main = do
  conn <- createDB
  createTables conn
  makeTypeTable conn
  makeTests conn
  putStrLn "tests table:"
  r <- query_ conn "SELECT * from tests" :: IO [TestField]
  mapM_ print r
  putStrLn "types table:"
  r <- query_ conn "SELECT * from template_types" :: IO [TestField]
  mapM_ print r
  close conn
