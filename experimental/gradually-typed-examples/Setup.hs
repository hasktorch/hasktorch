import Data.List (isPrefixOf)
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
      pure emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      mTokenizersDir <- ensureTokenizers
      case mTokenizersDir of
        Nothing -> do
          putStrLn "libtokenizers not found, skipping configuration."
          confHook simpleUserHooks (gpd, hbi) flags
        Just tokenizersDir -> do
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

getLocalUserTokenizersDir :: IO FilePath
getLocalUserTokenizersDir = do
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

ensureTokenizers :: IO (Maybe FilePath)
ensureTokenizers = do
  isSandbox <- isNixSandbox
  if isSandbox
    then return Nothing
    else download

isNixSandbox :: IO Bool
isNixSandbox = do
  nix <- lookupEnv "NIX_BUILD_TOP"
  case nix of
    Just path -> do
      let isNixPath = any (`isPrefixOf` path) ["/build", "/private/tmp/nix-build"]
      if isNixPath
        then do
          putStrLn "Nix sandbox detected; skipping libtorch download."
          return True
        else do
          return False
    Nothing -> return False

download :: IO (Maybe FilePath)
download = do
  skip   <- lookupEnv "TOKENIZERS_SKIP_DOWNLOAD"
  case skip of
    Just _ -> do
      putStrLn "TOKENIZERS_SKIP_DOWNLOAD set; assuming libtokenizers exists globally."
      return Nothing
    Nothing -> do
      dest   <- getLocalUserTokenizersDir
      let marker = dest </> ".ok"
      exists <- doesFileExist marker
      present<- doesDirectoryExist dest
      if present && exists
        then pure $ Just dest
        else do
          putStrLn $ "tokenizers not found, installing to " <> dest
          downloadAndExtractTokenizersTo dest
          writeFile marker ""
          pure $ Just dest

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
    let unpacked = tmpDir </> "libtokenizers"
    exists <- doesDirectoryExist unpacked
    let src = if exists then unpacked else tmpDir
    -- We want to move the directory since this operation is atomic.
    -- If that doesn't work we fall back to copying.
    (renameDirectory src dest) `catch` (\(_::IOException) -> copyTree src dest)
    putStrLn "tokenizers extracted successfully (global cache)."


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

addRPath :: FilePath -> ProgramDb -> ProgramDb
addRPath libDir progDb =
  userSpecifyArgs (programName ldProgram)
  ["-Wl,-rpath," ++ libDir]
  progDb
