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
import Control.Monad.Catch (MonadThrow)
import Data.Int (Int16)
import Data.Singletons (SingI (..), SingKind (..))
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Prelude (forgetIsChecked, pattern Demoted')
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Type as ATen

data Generator (device :: Device (DeviceType Nat)) where
  UnsafeGenerator ::
    forall device.
    { generatorSeed :: Word64,
      generatorDeviceType :: DeviceType Int16,
      generatorState :: TVar (Maybe (ForeignPtr ATen.Generator))
    } ->
    Generator device

type role Generator nominal

sMkGenerator ::
  forall m device.
  MonadThrow m =>
  -- | generator device singleton
  SDevice device ->
  -- | initial seed
  Word64 ->
  -- | returned generator
  m (Generator device)
sMkGenerator generatorDevice generatorSeed =
  unsafeThrowableIO $
    let generatorDeviceType = forgetIsChecked . fromSing $ generatorDevice
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
  forall m device.
  (SingI device, MonadThrow m) =>
  -- | initial seed
  Word64 ->
  -- | returned generator
  m (Generator device)
mkGenerator = sMkGenerator (sing @device)

sGeneratorToDevice ::
  forall m generatorDevice' generatorDevice.
  MonadThrow m =>
  SDevice generatorDevice' ->
  Generator generatorDevice ->
  m (Generator generatorDevice')
sGeneratorToDevice (Demoted' generatorDeviceType') UnsafeGenerator {..}
  | generatorDeviceType' == generatorDeviceType =
    pure $ UnsafeGenerator generatorSeed generatorDeviceType' generatorState
sGeneratorToDevice device' UnsafeGenerator {..} =
  sMkGenerator device' generatorSeed

generatorToDevice ::
  forall m generatorDevice' generatorDevice.
  (SingI generatorDevice', MonadThrow m) =>
  Generator generatorDevice ->
  m (Generator generatorDevice')
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
