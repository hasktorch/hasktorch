{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Maybe (fromMaybe)
import qualified Data.ByteString.Lazy as B
import qualified Network.Browser as BR
import Network.HTTP (rspCode, rspBody)
import Network.URI (parseURI)

downloadSite :: FilePath
downloadSite = "http://yann.lecun.com/exdb/mnist/"

download :: FilePath -> String -> IO ()
download outDir downloadFile = do
  let outFile = outDir ++ downloadFile
  let url = downloadSite ++ downloadFile
  let uri = fromMaybe (error ("Invalid URI " ++ url))
            (parseURI url)
  (_, result) <- BR.browse $ do
    BR.setMaxErrorRetries $ Just 3
    BR.setAllowRedirects True
    BR.request $ BR.defaultGETRequest_ uri
  case (rspCode result) of
    (2, 0, 0) -> B.writeFile outFile (rspBody result)
    s -> error ( "Error: " ++ show s ++ " " ++ url)

run :: FilePath -> IO ()
run dir = mapM_ (download dir) [
  "train-labels-idx1-ubyte.gz",
  "train-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
  "t10k-images-idx3-ubyte.gz"
  ]

main :: IO ()
main = do
  run "./"
  putStrLn "Finished downloading"
