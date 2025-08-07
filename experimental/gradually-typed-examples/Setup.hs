{-# LANGUAGE OverloadedStrings #-}
import Distribution.Simple
import Distribution.Simple.Program
import Distribution.Simple.Setup
import Distribution.Simple.LocalBuildInfo
import Distribution.Types.LocalBuildInfo
import Distribution.Types.GenericPackageDescription
import Distribution.Types.HookedBuildInfo
import Distribution.PackageDescription
import Distribution.System
import System.Directory
import System.FilePath
import System.Process
import System.IO.Temp
import Control.Monad
import Network.HTTP.Simple
import qualified Data.ByteString.Lazy as LBS
import System.Environment (lookupEnv)
import Data.Maybe (fromMaybe)
import Control.Exception
import GHC.IO.Exception (IOException)

main :: IO ()
main = defaultMainWithHooks $ simpleUserHooks
  { preConf = \_ _ -> do
      _ <- ensureTokenizers
      pure emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      tokenizersDir <- getGlobalTokenizersDir
      let tokLibDir    = tokenizersDir </> "lib"
          tokInclude   = tokenizersDir </> "include"
      let updatedFlags = flags
            { configExtraLibDirs     = tokLibDir : configExtraLibDirs flags
            , configExtraIncludeDirs = tokInclude : configExtraIncludeDirs flags
            }
      lbi <- confHook simpleUserHooks (gpd, hbi) updatedFlags
      return $ case buildOS of
        OSX   -> lbi { withPrograms = addRPath tokLibDir (withPrograms lbi) }
        Linux -> lbi { withPrograms = addRPath tokLibDir (withPrograms lbi) }
        _     -> lbi
  }

-- === tokenizers settings ===
tokenizersVersion :: String
tokenizersVersion = "v0.1"

getGlobalTokenizersDir :: IO FilePath
getGlobalTokenizersDir = do
  mHome <- lookupEnv "TOKENIZERS_HOME"
  base  <- case mHome of
    Just h  -> pure h
    Nothing -> getXdgDirectory XdgCache "tokenizers"
  pure $ base </> tokenizersVersion </> platformTag

platformTag :: FilePath
platformTag = case (buildOS, buildArch) of
  (OSX,    AArch64) -> "macos-arm64"
  (OSX,    X86_64)  -> "macos-x86_64"
  (Linux,  X86_64)  -> "linux-x86_64"
  _ -> error $ "Unsupported platform: " <> show (buildOS, buildArch)

-- === tokenizers download ===
ensureTokenizers :: IO FilePath
ensureTokenizers = do
  skip   <- lookupEnv "TOKENIZERS_SKIP_DOWNLOAD"
  if skip /= Nothing then getGlobalTokenizersDir else do
    dest   <- getGlobalTokenizersDir
    let marker = dest </> ".ok"
    exists <- doesFileExist marker
    present<- doesDirectoryExist dest
    if present && exists then pure dest else do
      putStrLn $ "tokenizers not found, installing to " <> dest
      downloadAndExtractTokenizersTo dest
      writeFile marker ""
      pure dest

downloadAndExtractTokenizersTo :: FilePath -> IO ()
downloadAndExtractTokenizersTo dest = do
  createDirectoryIfMissing True dest
  (url, fname) <- computeTokenizersURL
  putStrLn $ "Downloading tokenizers from: " ++ url
  withSystemTempDirectory "tokenizers-download" $ \tmpDir -> do
    let outPath = tmpDir </> fname
    req  <- parseRequest url
    res  <- httpLBS req
    LBS.writeFile outPath (getResponseBody res)
    putStrLn "Extracting tokenizers..."
    callProcess "unzip" ["-q", outPath, "-d", tmpDir]
    let src = if doesDirectoryExist (tmpDir </> "libtokenizers-") then head (filter ("libtokenizers-" `isPrefixOf`) <$> listDirectory tmpDir) else tmpDir
    (renameDirectory (tmpDir </> src) dest) `catch` \(_::IOException) -> copyTree (tmpDir </> src) dest
    putStrLn "tokenizers installed."

computeTokenizersURL :: IO (String, String)
computeTokenizersURL = do
  let v = tokenizersVersion
      arch = case buildOS of
        OSX   -> if buildArch == AArch64 then "macos" else "macos"
        Linux -> "linux"
        _     -> error "Unsupported platform for tokenizers"
      fileName = "libtokenizers-" ++ arch ++ ".zip"
      url      = "https://github.com/hasktorch/tokenizers/releases/download/libtokenizers-" ++ v ++ "/" ++ fileName
  pure (url, fileName)

copyTree :: FilePath -> FilePath -> IO ()
copyTree src dest = do
  createDirectoryIfMissing True dest
  entries <- listDirectory src
  forM_ entries $ \e -> do
    let s = src </> e
        d = dest </> e
    isDir <- doesDirectoryExist s
    if isDir then copyTree s d else copyFile s d

