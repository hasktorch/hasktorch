{-# LANGUAGE CPP #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Data.Char (toLower)
import Data.List (isPrefixOf, isInfixOf, nubBy)
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
import qualified Data.ByteString.Lazy.Char8 as LBSC
import System.Environment (lookupEnv)
import Data.Maybe (fromMaybe, listToMaybe)
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
          putStrLn "libtorch not found or handled by Nix, skipping configuration."
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
          
          -- Add RPath so the binary finds the libs at runtime
          case buildOS of
            OSX -> return $ lbi { withPrograms = addRPath libDir $ addLdClassicFlag (withPrograms lbi) }
            Linux -> return $ lbi { withPrograms = addRPath libDir (withPrograms lbi) }
            _ -> return $ lbi
  }

getLibtorchVersion :: IO String
getLibtorchVersion = do
  mVersion <- lookupEnv "LIBTORCH_VERSION"
  case mVersion of
    Nothing -> return "2.9.1"
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
  fromMaybe "cpu" <$> lookupEnv "LIBTORCH_CUDA_VERSION"  -- "cpu" | "cu121" | "cu118"

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

          -- Download NVIDIA libs if on Linux + CUDA
          flavor <- getCudaFlavor
          when (buildOS == Linux && "cu" `isPrefixOf` flavor) $ do
             -- Dynamically detect which NVIDIA library versions are needed
             -- and extract them to dest/lib (same directory as libtorch_cuda.so)
             downloadNvidiaLibs dest

          -- Create an idempotence marker
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

-- | Downloads necessary NVIDIA wheels from PyPI and extracts libs to dest/lib
-- Dynamically detects which library versions are needed by inspecting libtorch
-- Uses iterative detection to find transitive dependencies
downloadNvidiaLibs :: FilePath -> IO ()
downloadNvidiaLibs dest = do
  let libDir = dest </> "lib"

  -- Pass 1: Detect and download direct dependencies from libtorch
  putStrLn "Pass 1: Detecting NVIDIA library dependencies from libtorch..."
  neededLibs <- detectNeededNvidiaLibs libDir

  if null neededLibs
    then putStrLn "No missing NVIDIA libraries detected (all bundled with libtorch)."
    else do
      putStrLn $ "Found " ++ show (length neededLibs) ++ " NVIDIA libraries to download:"
      forM_ neededLibs $ \(lib, ver) ->
        putStrLn $ "  - " ++ lib ++ " (version " ++ ver ++ ")"
      downloadAndExtractPackages dest neededLibs

      -- Pass 2: Check downloaded libraries for transitive dependencies
      putStrLn "\nPass 2: Checking downloaded libraries for transitive dependencies..."
      transitiveLibs <- detectNeededNvidiaLibs libDir

      if null transitiveLibs
        then putStrLn "No additional transitive dependencies found."
        else do
          putStrLn $ "Found " ++ show (length transitiveLibs) ++ " additional libraries to download:"
          forM_ transitiveLibs $ \(lib, ver) ->
            putStrLn $ "  - " ++ lib ++ " (version " ++ ver ++ ")"
          downloadAndExtractPackages dest transitiveLibs

-- | Detect which NVIDIA libraries are needed by inspecting libtorch's dependencies
-- Returns list of (library name, major version) tuples, e.g., [("cusparse", "12"), ("cufft", "11")]
detectNeededNvidiaLibs :: FilePath -> IO [(String, String)]
detectNeededNvidiaLibs libDir = do
  putStrLn $ "  Inspecting libraries in: " ++ libDir

  -- Libraries we care about (including transitive dependencies)
  let targetLibs = ["cusparse", "cufft", "curand", "cublas", "cusolver", "nvjitlink"]

  -- Check libtorch libraries first
  let libtorchFiles = ["libtorch_cuda.so", "libtorch.so"]

  -- Also check any NVIDIA libraries that were already downloaded (for transitive deps)
  libDirContents <- listDirectory libDir
  let nvidiaLibs = filter (\f -> any (\target -> ("lib" ++ target) `isPrefixOf` f && ".so" `isInfixOf` f) targetLibs) libDirContents

  let soFiles = libtorchFiles ++ nvidiaLibs

  needed <- forM soFiles $ \soFile -> do
    let fullPath = libDir </> soFile
    exists <- doesFileExist fullPath
    if not exists
      then do
        putStrLn $ "  Skipping " ++ soFile ++ " (not found)"
        return []
      else do
        putStrLn $ "  Analyzing dependencies of " ++ soFile ++ "..."
        deps <- extractNvidiaDeps fullPath targetLibs
        forM_ deps $ \(lib, ver) ->
          putStrLn $ "    Found dependency: lib" ++ lib ++ ".so." ++ ver
        return deps

  -- Flatten results and remove duplicates
  let allNeeded = concat needed
      uniqueNeeded = nubBy (\(a,_) (b,_) -> a == b) allNeeded

  putStrLn $ "  Total unique NVIDIA dependencies found: " ++ show (length uniqueNeeded)

  -- Filter out libraries that are already bundled
  filterM (notBundled libDir) uniqueNeeded

-- | Check if a library is already bundled in libtorch
notBundled :: FilePath -> (String, String) -> IO Bool
notBundled libDir (libName, _ver) = do
  files <- listDirectory libDir
  -- Match exactly "libNAME.so" or "libNAME-*.so" to avoid false matches
  -- (e.g., "libcusparse" should not match "libcusparseLt")
  let pattern = "lib" ++ libName
  let bundled = any (\f ->
        (f == pattern ++ ".so" ||
         (pattern ++ ".so.") `isPrefixOf` f ||
         (pattern ++ "-") `isPrefixOf` f && ".so" `isInfixOf` f)) files
  return (not bundled)

-- | Extract NVIDIA library dependencies using readelf
extractNvidiaDeps :: FilePath -> [String] -> IO [(String, String)]
extractNvidiaDeps soFile targetLibs = do
  -- Try readelf first
  result <- tryReadelf soFile
  case result of
    Just output -> return $ parseNvidiaDeps output targetLibs
    Nothing -> do
      -- Fallback: try objdump
      result2 <- tryObjdump soFile
      case result2 of
        Just output -> return $ parseNvidiaDeps output targetLibs
        Nothing -> return []

tryReadelf :: FilePath -> IO (Maybe String)
tryReadelf soFile = do
  result <- try $ readProcess "readelf" ["-d", soFile] ""
  case result of
    Right output -> return (Just output)
    Left (_ :: IOException) -> return Nothing

tryObjdump :: FilePath -> IO (Maybe String)
tryObjdump soFile = do
  result <- try $ readProcess "objdump" ["-p", soFile] ""
  case result of
    Right output -> return (Just output)
    Left (_ :: IOException) -> return Nothing

-- | Parse readelf/objdump output to extract NVIDIA library dependencies
-- Example: "libcusparse.so.12" -> ("cusparse", "12")
-- Case-insensitive matching to handle libraries like "libnvJitLink" vs "libnvjitlink"
parseNvidiaDeps :: String -> [String] -> [(String, String)]
parseNvidiaDeps output targetLibs =
  [ (lib, ver)
  | line <- lines output
  , lib <- targetLibs
  , let pattern = "lib" ++ lib ++ ".so."
  , let lineLower = map toLower line
  , let patternLower = map toLower pattern
  , patternLower `isInfixOf` lineLower
  , let ver = extractVersion line lib
  , not (null ver)
  ]

extractVersion :: String -> String -> String
extractVersion line libName =
  case findSubstring pattern line of
    Just idx ->
      let afterPattern = drop (idx + length pattern) line
      in takeWhile isDigit afterPattern
    Nothing -> ""
  where
    pattern = "lib" ++ libName ++ ".so."
    isDigit c = c >= '0' && c <= '9'

    -- Case-insensitive substring search
    findSubstring :: String -> String -> Maybe Int
    findSubstring needle haystack = go 0 haystack
      where
        needleLower = map toLower needle
        go _ [] = Nothing
        go idx str
          | length str >= length needle &&
            map toLower (take (length needle) str) == needleLower = Just idx
          | otherwise = go (idx + 1) (tail str)

downloadAndExtractPackages :: FilePath -> [(String, String)] -> IO ()
downloadAndExtractPackages dest neededLibs =
  withSystemTempDirectory "nvidia-libs-download" $ \tmpDir -> do
    forM_ neededLibs $ \(libName, majorVer) -> do
      -- Special case: nvidia-cufft-cu12 provides .so.11, but we need .so.12
      -- In this case, use the generic nvidia-cufft package which has 12.x
      let useGenericFirst = (libName == "cufft" && majorVer == "12")

      -- Map major version to PyPI suffix (12 -> cu12, 11 -> cu11, etc.)
      let pypiSuffix = "cu" ++ majorVer
      let pkgWithSuffix = "nvidia-" ++ libName ++ "-" ++ pypiSuffix
      let pkgGeneric = "nvidia-" ++ libName

      -- Try generic first for known mismatches, otherwise try CUDA-specific first
      (finalPkg, finalUrl) <- if useGenericFirst
        then do
          putStrLn $ "Fetching metadata for " ++ pkgGeneric ++ " (known version mismatch)..."
          mUrl <- getPyPiWheelUrl pkgGeneric
          case mUrl of
            Just url -> return (pkgGeneric, Just url)
            Nothing -> do
              putStrLn $ "  Not found, trying: " ++ pkgWithSuffix ++ "..."
              url <- getPyPiWheelUrl pkgWithSuffix
              return (pkgWithSuffix, url)
        else do
          putStrLn $ "Fetching metadata for " ++ pkgWithSuffix ++ "..."
          mUrl <- getPyPiWheelUrl pkgWithSuffix
          -- Fallback to generic package name if versioned one not found
          -- (e.g., nvidia-curand instead of nvidia-curand-cu10)
          case mUrl of
            Just url -> return (pkgWithSuffix, Just url)
            Nothing -> do
              putStrLn $ "  Not found, trying fallback: " ++ pkgGeneric ++ "..."
              url <- getPyPiWheelUrl pkgGeneric
              return (pkgGeneric, url)

      case finalUrl of
        Nothing -> putStrLn $ "Warning: Could not find manylinux wheel for " ++ finalPkg
        Just url -> do
          let fileName = takeFileName url
          let downloadPath = tmpDir </> fileName
          putStrLn $ "Downloading " ++ fileName ++ "..."
          
          request  <- parseRequest url
          response <- httpLBS request
          LBS.writeFile downloadPath (getResponseBody response)
          
          putStrLn $ "Extracting libraries from " ++ fileName ++ "..."
          archive <- toArchive <$> LBS.readFile downloadPath
          
          -- Iterate over entries and extract only .so files to dest/lib
          let libDest = dest </> "lib"
          createDirectoryIfMissing True libDest
          
          forM_ (zEntries archive) $ \entry -> do
            let path = eRelativePath entry
                isSharedObj = ".so" `isInfixOf` path
                -- Wheels put libs in nvidia/<pkg>/lib/ or lib/
                isLibDir = "lib/" `isInfixOf` path 
            
            when (isSharedObj && isLibDir) $ do
              let entryFileName = takeFileName path
              let targetPath = libDest </> entryFileName
              putStrLn $ "  Extracting: " ++ entryFileName ++ " -> " ++ targetPath
              -- Write the file
              let entryData = fromEntry entry
              LBS.writeFile targetPath entryData
              -- Set readable and executable permissions for shared libraries
              -- Using only owner permissions (older directory library compatibility)
              let perms = foldl (flip ($)) emptyPermissions
                          [ setOwnerReadable True
                          , setOwnerExecutable True
                          ]
              setPermissions targetPath perms

-- | Quick and dirty PyPI JSON parser to find manylinux x86_64 url
-- Avoids adding Aeson dependency to Setup.hs
getPyPiWheelUrl :: String -> IO (Maybe String)
getPyPiWheelUrl pkg = do
  let jsonUrl = "https://pypi.org/pypi/" ++ pkg ++ "/json"
  request <- parseRequest jsonUrl
  response <- httpLBS request
  let body = getResponseBody response
  
  -- We look for the "url" field inside an object that also has "manylinux" and "x86_64" in the filename
  -- This is a heuristic search on the raw JSON string
  let urls = extractUrls body
  return $ listToMaybe [ u | u <- urls, "manylinux" `isInfixOf` u, "x86_64" `isInfixOf` u ]

-- | Helper to extract all strings that look like URLs from JSON
extractUrls :: LBS.ByteString -> [String]
extractUrls content = 
  let parts = LBSC.split '"' content
      -- Filter for https URLs
  in [ LBSC.unpack p | p <- parts, "https://" `LBSC.isPrefixOf` p, ".whl" `LBSC.isSuffixOf` p ]

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
      "cpu"  -> ( "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-" ++ v ++ "%2Bcpu.zip"
                , "libtorch-linux.zip" )
      cudaVersion -> ( "https://download.pytorch.org/libtorch/" ++ cudaVersion ++"/libtorch-shared-with-deps-" ++ v ++ "%2B" ++ cudaVersion ++ ".zip"
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
  [ "-Wl,-rpath," ++ libDir          -- for runtime library search
  ]
  progDb
