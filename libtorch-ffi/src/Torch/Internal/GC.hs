{-# LANGUAGE CPP #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Internal.GC where

import Control.Concurrent (threadDelay)
import Control.Concurrent.Async
import Control.Exception.Safe (Exception, MonadThrow, Typeable, catch, throwIO, throwM)
import Control.Monad (when)
import Data.List (isPrefixOf)
import Foreign.C.Types
import GHC.ExecutionStack
import Language.C.Inline.Cpp.Exceptions
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC)
import System.SysInfo

foreign import ccall unsafe "hasktorch_finalizer.h showWeakPtrList"
  c_showWeakPtrList :: CInt -> IO ()

-- malloc_trim is a glibc function. It doesn't exist on macos.
#ifdef ENABLE_DUMMY_MALLOC_TRIM
mallocTrim :: CInt -> IO ()
mallocTrim _ = return ()
#else
foreign import ccall unsafe "malloc.h malloc_trim"
  mallocTrim :: CInt -> IO ()
#endif

-- | Returns all objects of libtorch.
-- Each time it is called, the age of the object increases by one.
-- Dumps objects that are greater than or equal to the argument of age.
dumpLibtorchObjects ::
  -- | age
  Int ->
  -- | output
  IO ()
dumpLibtorchObjects age = c_showWeakPtrList (fromIntegral age)

newtype HasktorchException = HasktorchException String
  deriving (Show)

instance Exception HasktorchException

unsafeThrowableIO :: forall a m. MonadThrow m => IO a -> m a
unsafeThrowableIO a = unsafePerformIO $ (pure <$> a) `catch` (\(CppStdException msg) -> pure . throwM $ HasktorchException msg)

prettyException :: IO a -> IO a
prettyException func =
  func `catch` \a@(CppStdException message) -> do
    flag <- lookupEnv "HASKTORCH_DEBUG"
    when (flag /= Just "0") $ do
      mst <- showStackTrace
      case mst of
        Just st -> hPutStrLn stderr st
        Nothing -> hPutStrLn stderr "Cannot show stacktrace"
      hPutStrLn stderr message
    throwIO a
{-# INLINE prettyException #-}

retryWithGC' :: Int -> IO a -> IO a
retryWithGC' count func =
  func `catch` \a@(CppStdException message) ->
    if isPrefixOf msgOutOfMemory message
      then
        if count <= 0
          then throwIO $ userError $ "Too many calls to performGC, " ++ message
          else do
            performGC
            mallocTrim 0
            threadDelay 1000 -- We need delta delay(1ms) to wait GC.
            retryWithGC' (count -1) func
      else throwIO a
  where
    msgOutOfMemory :: String
    msgOutOfMemory = "Exception: CUDA out of memory."
{-# INLINE retryWithGC' #-}

retryWithGC :: IO a -> IO a
retryWithGC func = prettyException $ retryWithGC' 10 func
{-# INLINE retryWithGC #-}

checkOSMemoryWithGC :: IO ()
checkOSMemoryWithGC = do
  v <- sysInfo
  case v of
    Right stat -> do
      let rate = (fromIntegral (freeram stat) / fromIntegral (totalram stat))
      if rate <= 0.5
        then do
          performGC
          mallocTrim 0
        else return ()
    Left _ -> return ()
  threadDelay (500 * 1000) -- wait 500msec
  checkOSMemoryWithGC

monitorMemory :: IO () -> IO ()
monitorMemory func = do
  func `race` checkOSMemoryWithGC
  return ()
