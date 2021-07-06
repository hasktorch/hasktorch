{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Random (Generator, mkGenerator, sMkGenerator, withGenerator) where

import Control.Concurrent.STM (TVar, atomically, newTVarIO, readTVar, writeTVar)
import Data.Int (Int16)
import Data.Singletons (SingI (..), SingKind (..))
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
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
  forall device.
  SDevice device ->
  Word64 ->
  IO (Generator device)
sMkGenerator device generatorSeed =
  let generatorDeviceType = forgetIsChecked . fromSing $ device
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
  Word64 ->
  IO (Generator device)
mkGenerator = sMkGenerator (sing @device)

withGenerator ::
  forall requiresGradient layout device dataType shape device'.
  (ForeignPtr ATen.Generator -> IO (Tensor requiresGradient layout device dataType shape)) ->
  Tensor requiresGradient layout device dataType shape ->
  Generator device' ->
  (Tensor requiresGradient layout device dataType shape, Generator (device <+> device'))
withGenerator f _ UnsafeGenerator {..} = unsafePerformIO $ do
  mGenPtr <- atomically $ do
    mGenPtr <- readTVar generatorState
    case mGenPtr of
      Just genPtr -> do
        writeTVar generatorState Nothing
        return $ Just genPtr
      Nothing -> return Nothing
  genPtr <- case mGenPtr of
    Just genPtr -> return genPtr
    Nothing -> case generatorDeviceType of
      CPU -> ATen.newCPUGenerator generatorSeed
      CUDA deviceId -> do
        genPtr <- ATen.newCUDAGenerator (fromIntegral deviceId)
        ATen.generator_set_current_seed genPtr generatorSeed
        return genPtr
  a <- f genPtr
  nextSeed <- ATen.generator_current_seed genPtr
  nextGen <- newTVarIO (Just genPtr)
  return (a, UnsafeGenerator nextSeed generatorDeviceType nextGen)
