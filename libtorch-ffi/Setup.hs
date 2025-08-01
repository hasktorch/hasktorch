import Distribution.Simple
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

-- Configuration
libtorchVersion :: String
libtorchVersion = "2.5.0"

main :: IO ()
main = defaultMainWithHooks $ simpleUserHooks
  { preConf = \args flags -> do
      -- Download libtorch if needed before configuration
      ensureLibtorch
      return emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      -- Get the libtorch directory
      let libtorchDir = "libtorch"
          libDir = libtorchDir </> "lib"
          includeDir = libtorchDir </> "include"
      
      -- Update the flags with libtorch paths
      let updatedFlags = flags
            { configExtraLibDirs = libDir : configExtraLibDirs flags
            , configExtraIncludeDirs = includeDir : (includeDir </> "torch" </> "csrc" </> "api" </> "include") : configExtraIncludeDirs flags
            }
      
      -- Call the default configuration hook with updated flags
      confHook simpleUserHooks (gpd, hbi) updatedFlags
  }

ensureLibtorch :: IO ()
ensureLibtorch = do
  -- Check if user wants to skip download
  skipDownload <- lookupEnv "LIBTORCH_SKIP_DOWNLOAD"
  case skipDownload of
    Just _ -> putStrLn "LIBTORCH_SKIP_DOWNLOAD is set, skipping libtorch download"
    Nothing -> do
      libtorchExists <- doesDirectoryExist "libtorch"
      
      unless libtorchExists $ do
        putStrLn "libtorch not found, downloading..."
        downloadAndExtractLibtorch
        
        -- Copy libraries to be bundled
        putStrLn "Setting up bundled libraries..."
        setupBundledLibraries

downloadAndExtractLibtorch :: IO ()
downloadAndExtractLibtorch = do
  -- Check for CUDA version from environment variable
  cudaVersion <- lookupEnv "LIBTORCH_CUDA_VERSION"
  let computeArch = fromMaybe "cpu" cudaVersion
  
  -- Determine OS and architecture
  let (url, fileName) = case buildOS of
        OSX -> case buildArch of
          AArch64 -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-" ++ libtorchVersion ++ ".zip"
                     , "libtorch-macos-arm64.zip" )
          X86_64 -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-" ++ libtorchVersion ++ ".zip"
                    , "libtorch-macos-x86_64.zip" )
          _ -> error "Unsupported macOS architecture"
        Linux -> case computeArch of
          "cpu" -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-" ++ libtorchVersion ++ "%2Bcpu.zip"
                   , "libtorch-linux.zip" )
          "cu117" -> ( "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-" ++ libtorchVersion ++ "%2Bcu117.zip"
                     , "libtorch-linux-cu117.zip" )
          "cu118" -> ( "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-" ++ libtorchVersion ++ "%2Bcu118.zip"
                     , "libtorch-linux-cu118.zip" )
          "cu121" -> ( "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-" ++ libtorchVersion ++ "%2Bcu121.zip"
                     , "libtorch-linux-cu121.zip" )
          _ -> error $ "Unsupported CUDA version: " ++ computeArch ++ ". Use cpu, cu117, cu118, or cu121"
        Windows -> error "Windows is not yet supported by this setup"
        _ -> error $ "Unsupported operating system: " ++ show buildOS
  
  putStrLn $ "Downloading libtorch from: " ++ url
  
  withSystemTempDirectory "libtorch-download" $ \tmpDir -> do
    let downloadPath = tmpDir </> fileName
    
    -- Download the file
    request <- parseRequest url
    response <- httpLBS request
    LBS.writeFile downloadPath (getResponseBody response)
    
    putStrLn "Download complete. Extracting..."
    
    -- Extract the archive
    callProcess "unzip" ["-q", downloadPath]
    
    putStrLn "libtorch extracted successfully"

setupBundledLibraries :: IO ()
setupBundledLibraries = do
  -- Create a directory for bundled libraries if it doesn't exist
  createDirectoryIfMissing True "cbits"
  
  -- List of core libraries to bundle
  let coreLibs = case buildOS of
        OSX -> [ "libc10.dylib"
               , "libtorch.dylib"
               , "libtorch_cpu.dylib"
               ]
        Linux -> [ "libc10.so"
                 , "libtorch.so"
                 , "libtorch_cpu.so"
                 ]
        _ -> error "Unsupported OS"
  
  -- Copy core libraries to cbits directory
  forM_ coreLibs $ \lib -> do
    let srcPath = "libtorch" </> "lib" </> lib
        dstPath = "cbits" </> lib
    
    exists <- doesFileExist srcPath
    when exists $ do
      putStrLn $ "Copying " ++ lib ++ " to cbits/"
      copyFile srcPath dstPath