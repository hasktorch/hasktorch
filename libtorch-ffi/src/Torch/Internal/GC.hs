{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Internal.GC where

import Control.Concurrent (threadDelay)
import Control.Concurrent.Async
import Control.Exception.Safe (catch, throwIO)
import Control.Monad (when)
import Data.List (isPrefixOf)
import GHC.ExecutionStack
import Language.C.Inline.Cpp.Exceptions (CppException (..))
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import System.Mem (performGC)
import System.SysInfo

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

retryWithGC' :: Int -> IO a -> IO a
retryWithGC' count func =
  func `catch` \a@(CppStdException message) ->
    if isPrefixOf msgOutOfMemory message
      then
        if count <= 0
          then throwIO $ CppStdException $ "Too many calls to performGC, " ++ message
          else do
            performGC
            threadDelay 1000 -- We need delta delay(1ms) to wait GC.
            retryWithGC' (count -1) func
      else throwIO a
  where
    msgOutOfMemory :: String
    msgOutOfMemory = "Exception: CUDA out of memory."

retryWithGC :: IO a -> IO a
retryWithGC func = prettyException $ retryWithGC' 10 func

checkOSMemoryWithGC :: IO ()
checkOSMemoryWithGC = do
  v <- sysInfo
  case v of
    Right stat -> do
      let rate = (fromIntegral (freeram stat) / fromIntegral (totalram stat))
      if rate <= 0.5
        then performGC
        else return ()
    Left _ -> return ()
  threadDelay (500 * 1000) -- wait 500msec
  checkOSMemoryWithGC

monitorMemory :: IO () -> IO ()
monitorMemory func = do
  func `race` checkOSMemoryWithGC
  return ()
