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

main :: IO ()
main = defaultMainWithHooks $ simpleUserHooks
  { preConf = \_ _ -> do
      _ <- ensureLibtorch 
      pure emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      libtorchDir <- getGlobalLibtorchDir
      let libDir     = libtorchDir </> "lib"
          includeDir = libtorchDir </> "include"
      
      let updatedFlags = flags
            { configExtraLibDirs      = libDir : configExtraLibDirs flags
            , configExtraIncludeDirs  =
                includeDir
                : (includeDir </> "torch" </> "csrc" </> "api" </> "include")
                : configExtraIncludeDirs flags
            }
      -- Call the default configuration hook with updated flags
      lbi <- confHook simpleUserHooks (gpd, hbi) updatedFlags
      -- For macOS, add the -ld_classic flag to the linker
      return $
        case buildOS of
          OSX -> lbi { withPrograms = addRPath libDir $ addLdClassicFlag (withPrograms lbi) }
          Linux -> lbi { withPrograms = addRPath libDir (withPrograms lbi) }
          _ -> lbi
  }

libtorchVersion :: String
libtorchVersion = "2.5.0"

getGlobalLibtorchDir :: IO FilePath
getGlobalLibtorchDir = do
  mHome <- lookupEnv "LIBTORCH_HOME"
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

ensureLibtorch :: IO FilePath
ensureLibtorch = do
  skip <- lookupEnv "LIBTORCH_SKIP_DOWNLOAD"
  case skip of
    Just _ -> do
      putStrLn "LIBTORCH_SKIP_DOWNLOAD set; assuming libtorch exists globally."
      getGlobalLibtorchDir
    Nothing -> do
      dest <- getGlobalLibtorchDir
      let marker = dest </> ".ok"
      exists <- doesFileExist marker
      present <- doesDirectoryExist dest
      if present && exists
        then pure dest
        else do
          putStrLn $ "libtorch not found in global cache, installing to " <> dest
          downloadAndExtractLibtorchTo dest
          -- Create an idempotence marker that checks
          -- if we've already downloaded torch.
          -- Since we'll be moving everything this will
          -- be the our main reference.
          writeFile marker ""
          pure dest

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
    callProcess "unzip" ["-q", downloadPath, "-d", tmpDir]
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
  let v = libtorchVersion
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
      "cu117"-> ( "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-" ++ v ++ "%2Bcu117.zip"
                , "libtorch-linux-cu117.zip" )
      "cu118"-> ( "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-" ++ v ++ "%2Bcu118.zip"
                , "libtorch-linux-cu118.zip" )
      "cu121"-> ( "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-" ++ v ++ "%2Bcu121.zip"
                , "libtorch-linux-cu121.zip" )
      _      -> error $ "Unsupported CUDA version: " ++ flavor
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
