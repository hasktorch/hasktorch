{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Jit where

import Torch.Script
import Torch.Tensor
import Torch.NN
import Control.Concurrent.STM.TVar
import Control.Concurrent.STM (atomically)
import System.IO.Unsafe (unsafePerformIO)

newtype ScriptCache = ScriptCache { unScriptCache :: TVar (Maybe ScriptModule) }

newScriptCache :: IO ScriptCache
newScriptCache = ScriptCache <$> newTVarIO Nothing

jitIO :: ScriptCache -> ([Tensor] -> IO [Tensor]) -> [Tensor] -> IO [Tensor]
jitIO (ScriptCache cache) func input = do
  v <- readTVarIO cache
  script <- case v of
    Just script' -> return script'
    Nothing -> do
      m <- trace "MyModule" "forward" func input
      script' <- toScriptModule m
      atomically $ writeTVar cache (Just script')
      return script'
  IVTensor r0 <- forwardStoch script (map IVTensor input)
  return [r0]

jit :: ScriptCache -> ([Tensor] -> [Tensor]) -> [Tensor] -> [Tensor]
jit cache func input = unsafePerformIO $ jitIO cache (return . func) input
