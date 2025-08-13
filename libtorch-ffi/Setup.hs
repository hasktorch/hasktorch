{-# LANGUAGE CPP #-}

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
import GHC.IO.Exception
import Control.Exception
import Codec.Archive.Zip

#if MIN_VERSION_Cabal(3,14,0)
import Distribution.Utils.Path (makeSymbolicPath)
#else

makeSymbolicPath :: a -> a
makeSymbolicPath = id
#endif

main :: IO ()
main = defaultMainWithHooks $ simpleUserHooks
  { preConf = \_ _ -> do
      pure emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      mlibtorchDir <- ensureLibtorch
      case mlibtorchDir of
        Nothing -> do
          putStrLn "libtorch not found, skipping configuration."
          lbi <- confHook simpleUserHooks (gpd, hbi) flags
          -- For macOS, add the -ld_classic flag to the linker
          case buildOS of
            OSX -> return $ lbi { withPrograms = addLdClassicFlag (withPrograms lbi) }
            _ -> return $ lbi
        Just libtorchDir -> do
          libtorchDir <- getLocalUserLibtorchDir
          let libDir     = libtorchDir </> "lib"
              includeDir = libtorchDir </> "include"

          let updatedFlags = flags
                { configExtraLibDirs      = makeSymbolicPath libDir : configExtraLibDirs flags
                , configExtraIncludeDirs  =
                    makeSymbolicPath includeDir
                    : makeSymbolicPath (includeDir </> "torch" </> "csrc" </> "api" </> "include")
                    : configExtraIncludeDirs flags
                }
          -- Call the default configuration hook with updated flags
          lbi <- confHook simpleUserHooks (gpd, hbi) updatedFlags
          -- For macOS, add the -ld_classic flag to the linker
          case buildOS of
            OSX -> return $ lbi { withPrograms = addRPath libDir $ addLdClassicFlag (withPrograms lbi) }
            Linux -> return $ lbi { withPrograms = addRPath libDir (withPrograms lbi) }
            _ -> return $ lbi
  }

getLibtorchVersion :: IO String
getLibtorchVersion = do
  mVersion <- lookupEnv "LIBTORCH_VERSION"
  case mVersion of
    Nothing -> return "2.5.0"
    Just other -> return other

getLocalUserLibtorchDir :: IO FilePath
getLocalUserLibtorchDir = do
  mHome <- lookupEnv "LIBTORCH_HOME"
  libtorchVersion <- getLibtorchVersion
  base <- case mHome of
    Just h  -> pure h
    Nothing -> do
      -- XDG cache (Linux/macOS). Falls back to ~/.cache
      cache <- getXdgDirectory XdgCache "libtorch"
      pure cache
  flavor <- getCudaFlavor
  pure $ base </> libtorchVersion </> platformTag </> flavor

platformTag :: FilePath
platformTag =
  case (buildOS, buildArch) of
    (OSX,    AArch64) -> "macos-arm64"
    (OSX,    X86_64)  -> "macos-x86_64"
    (Linux,  X86_64)  -> "linux-x86_64"
    -- add more as needed
    _ -> error $ "Unsupported platform: " <> show (buildOS, buildArch)

getCudaFlavor :: IO String
getCudaFlavor = do
  fromMaybe "cpu" <$> lookupEnv "LIBTORCH_CUDA_VERSION"  -- "cpu" | "cu117" | "cu118" | "cu121"

ensureLibtorch :: IO (Maybe FilePath)
ensureLibtorch = do
  isSandbox <- isNixSandbox
  if isSandbox
    then return Nothing
    else downloadLibtorch

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

downloadLibtorch :: IO (Maybe FilePath)
downloadLibtorch = do
  skip <- lookupEnv "LIBTORCH_SKIP_DOWNLOAD"
  case skip of
    Just _ -> do
      putStrLn "LIBTORCH_SKIP_DOWNLOAD set; assuming libtorch exists globally."
      return Nothing
    Nothing -> do
      dest <- getLocalUserLibtorchDir
      let marker = dest </> ".ok"
      exists <- doesFileExist marker
      present <- doesDirectoryExist dest
      if present && exists
        then pure $ Just dest
        else do
          putStrLn $ "libtorch not found in local cache, installing to " <> dest
          downloadAndExtractLibtorchTo dest
          -- Create an idempotence marker that checks
          -- if we've already downloaded torch.
          -- Since we'll be moving everything this will
          -- be the our main reference.
          writeFile marker ""
          pure $ Just dest

downloadAndExtractLibtorchTo :: FilePath -> IO ()
downloadAndExtractLibtorchTo dest = do
  createDirectoryIfMissing True dest
  (url, fileName) <- computeURL
  putStrLn $ "Downloading libtorch from: " ++ url
  withSystemTempDirectory "libtorch-download" $ \tmpDir -> do
    let downloadPath = tmpDir </> fileName
    request  <- parseRequest url
    response <- httpLBS request
    LBS.writeFile downloadPath (getResponseBody response)
    putStrLn "Download complete. Extracting..."
    archive <- toArchive <$> LBS.readFile downloadPath
    extractFilesFromArchive [OptDestination tmpDir] archive
    let unpacked = tmpDir </> "libtorch"
    exists <- doesDirectoryExist unpacked
    let src = if exists then unpacked else tmpDir
    -- We want to move the directory since this operation is atomic.
    -- If that doesn't work we fall back to copying.
    (renameDirectory src dest) `catch` (\(_::IOException) -> copyTree src dest)
    putStrLn "libtorch extracted successfully (global cache)."

copyTree :: FilePath -> FilePath -> IO ()
copyTree src dest = do
  createDirectoryIfMissing True dest
  entries <- listDirectory src
  mapM_ (\e -> do
          let s = src  </> e
              d = dest </> e
          isDir <- doesDirectoryExist s
          if isDir then copyTree s d else copyFile s d
        ) entries

computeURL :: IO (String, String)
computeURL = do
  flavor <- getCudaFlavor
  v <- getLibtorchVersion
  pure $ case buildOS of
    OSX -> case buildArch of
      AArch64 -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-" ++ v ++ ".zip"
                 , "libtorch-macos-arm64.zip" )
      X86_64  -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-" ++ v ++ ".zip"
                 , "libtorch-macos-x86_64.zip" )
      _       -> error "Unsupported macOS arch"
    Linux -> case flavor of
      "cpu"  -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-" ++ v ++ "%2Bcpu.zip"
                , "libtorch-linux.zip" )
      cudaVersion -> ( "https://download.pytorch.org/libtorch/" ++ cudaVersion ++"/libtorch-cxx11-abi-shared-with-deps-" ++ v ++ "%2B" ++ cudaVersion ++ ".zip"
                , "libtorch-linux-" ++ cudaVersion ++ ".zip" )
    Windows -> error "Windows not supported by this setup"

-- Add -ld_classic flag to GHC program arguments for macOS
addLdClassicFlag :: ProgramDb -> ProgramDb
addLdClassicFlag progDb = 
  case lookupProgram ghcProgram progDb of
    Just ghc ->
      let ghc' = ghc { programOverrideArgs = ["-optl-ld_classic"] ++ programOverrideArgs ghc }
      in updateProgram ghc' progDb
    Nothing -> progDb

addRPath :: FilePath -> ProgramDb -> ProgramDb
addRPath libDir progDb =
  userSpecifyArgs (programName ldProgram)
  ["-Wl,-rpath," ++ libDir]
  progDb
