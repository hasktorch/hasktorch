{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Random where

import Control.Concurrent.STM (TVar, atomically, newTVarIO, readTVar, writeTVar)
import Data.Int (Int16)
import Data.Singletons (SingI (..), SingKind (..))
import Data.Singletons.Prelude.Check (pattern Demoted)
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Device (Device, DeviceType (..), SDevice)
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Type as ATen

data Generator (device :: Device) where
  UnsafeGenerator ::
    forall device.
    { generatorSeed :: Word64,
      generatorDeviceType :: DeviceType Int16,
      generatorState :: TVar (Maybe (ForeignPtr ATen.Generator))
    } ->
    Generator device

type role Generator nominal

sMkGenerator ::
  forall device.
  -- | generator device singleton
  SDevice device ->
  -- | initial seed
  Word64 ->
  -- | returned generator
  Generator device
sMkGenerator generatorDevice generatorSeed =
  unsafePerformIO $
    let generatorDeviceType = undefined . fromSing $ generatorDevice
     in case generatorDeviceType of
          CPU -> do
            genPtr <- ATen.newCPUGenerator generatorSeed
            generatorState <- newTVarIO (Just genPtr)
            return $ UnsafeGenerator {..}
          CUDA deviceId -> do
            genPtr <- ATen.newCUDAGenerator (fromIntegral deviceId)
            ATen.generator_set_current_seed genPtr generatorSeed
            generatorState <- newTVarIO (Just genPtr)
            return $ UnsafeGenerator {..}

mkGenerator ::
  forall device.
  SingI device =>
  -- | initial seed
  Word64 ->
  -- | returned generator
  Generator device
mkGenerator = sMkGenerator (sing @device)

sGeneratorToDevice ::
  forall generatorDevice' generatorDevice.
  SDevice generatorDevice' ->
  Generator generatorDevice ->
  Generator generatorDevice'
sGeneratorToDevice (Demoted generatorDeviceType') UnsafeGenerator {..}
  | generatorDeviceType' == generatorDeviceType =
    UnsafeGenerator generatorSeed generatorDeviceType' generatorState
sGeneratorToDevice device' UnsafeGenerator {..} =
  sMkGenerator device' generatorSeed

generatorToDevice ::
  forall generatorDevice' generatorDevice.
  SingI generatorDevice' =>
  Generator generatorDevice ->
  Generator generatorDevice'
generatorToDevice = sGeneratorToDevice (sing @generatorDevice')

withGenerator ::
  (ForeignPtr ATen.Generator -> IO (ForeignPtr ATen.Tensor)) ->
  Word64 ->
  DeviceType Int16 ->
  TVar (Maybe (ForeignPtr ATen.Generator)) ->
  IO (ForeignPtr ATen.Tensor, Word64, TVar (Maybe (ForeignPtr ATen.Generator)))
withGenerator f seed deviceType state = do
  mGenPtr <- atomically $ do
    mGenPtr <- readTVar state
    case mGenPtr of
      Just genPtr -> do
        writeTVar state Nothing
        return $ Just genPtr
      Nothing -> return Nothing
  genPtr <- case mGenPtr of
    Just genPtr -> return genPtr
    Nothing ->
      case deviceType of
        CPU -> ATen.newCPUGenerator seed
        CUDA deviceId -> do
          genPtr <- ATen.newCUDAGenerator (fromIntegral deviceId)
          ATen.generator_set_current_seed genPtr seed
          return genPtr
  t <- f genPtr
  nextSeed <- ATen.generator_current_seed genPtr
  nextGen <- newTVarIO (Just genPtr)
  return (t, nextSeed, nextGen)
