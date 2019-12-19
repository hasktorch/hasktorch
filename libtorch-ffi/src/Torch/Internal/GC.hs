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

import Control.Exception.Safe (catch,throwIO)
import Data.List (isPrefixOf)
import Language.C.Inline.Cpp.Exceptions (CppException(..))
import System.Mem (performGC)
import Control.Concurrent (threadDelay)
import Control.Concurrent.Async
import System.SysInfo

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
        retryWithGC' (count-1) func
    else throwIO a
  where
    msgOutOfMemory :: String
    msgOutOfMemory = "Exception: CUDA out of memory."

retryWithGC :: IO a -> IO a
retryWithGC = retryWithGC' 10

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
  threadDelay (500*1000) -- wait 500msec
  checkOSMemoryWithGC

monitorMemory :: IO () -> IO ()
monitorMemory func = do
  func `race` checkOSMemoryWithGC
  return ()
