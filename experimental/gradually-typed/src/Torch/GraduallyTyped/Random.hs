{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
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
import Control.Concurrent.STM.TVar (readTVarIO)
import Control.Monad.Catch (MonadThrow)
import Data.Int (Int16)
import Data.Proxy (Proxy (Proxy))
import Data.Singletons (SingI (..), SingKind (..))
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, Nat, natVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Internal.TensorOptions (TensorOptions (..), tensorDims, tensorOptions)
import Torch.GraduallyTyped.Prelude (forgetIsChecked, pattern Demoted')
import Torch.GraduallyTyped.Shape.Type (Dim)
import Torch.GraduallyTyped.Tensor.Type (Tensor (UnsafeTensor), TensorSpec (..), gitHubErrorMsg)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast0, cast1, cast2, cast4)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Type as ATen

newtype Generator (device :: Device (DeviceType Nat)) where
  UnsafeGenerator ::
    forall device.
    TVar (Either (SDevice device, Word64) (ForeignPtr ATen.Generator)) ->
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
sMkGenerator device seed =
  unsafeThrowableIO $ do
    let deviceType = forgetIsChecked . fromSing $ device
    genPtr <- case deviceType of
      CPU -> ATen.newCPUGenerator seed
      CUDA deviceId -> do
        genPtr <- ATen.newCUDAGenerator (fromIntegral deviceId)
        ATen.generator_set_current_seed genPtr seed
        pure genPtr
    UnsafeGenerator <$> newTVarIO (Right genPtr)

mkGenerator ::
  forall m device.
  (SingI device, MonadThrow m) =>
  -- | initial seed
  Word64 ->
  -- | returned generator
  m (Generator device)
mkGenerator = sMkGenerator (sing @device)

sSetGeneratorDevice ::
  forall m generatorDevice' generatorDevice.
  MonadThrow m =>
  SDevice generatorDevice' ->
  Generator generatorDevice ->
  m (Generator generatorDevice')
sSetGeneratorDevice = undefined

setGeneratorDevice ::
  forall m generatorDevice' generatorDevice.
  (SingI generatorDevice', MonadThrow m) =>
  Generator generatorDevice ->
  m (Generator generatorDevice')
setGeneratorDevice = sSetGeneratorDevice (sing @generatorDevice')

class SGetGeneratorDevice (device :: Device (DeviceType Nat)) where
  sGetGenPtrDevice ::
    ForeignPtr ATen.Generator ->
    SDevice device

instance SGetGeneratorDevice 'UncheckedDevice where
  sGetGenPtrDevice genPtr
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.generator_is_cuda genPtr) =
      case unsafePerformIO (cast1 ATen.generator_get_device genPtr) :: Int of
        deviceIndex -> SUncheckedDevice . CUDA . fromIntegral $ deviceIndex
    | otherwise = SUncheckedDevice CPU

instance SGetGeneratorDevice ('Device 'CPU) where
  sGetGenPtrDevice genPtr
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.generator_is_cuda genPtr) =
      error $
        "The generator should be on CPU but is on CUDA. "
          <> gitHubErrorMsg
    | otherwise = SDevice SCPU

instance KnownNat deviceIndex => SGetGeneratorDevice ('Device ('CUDA deviceIndex)) where
  sGetGenPtrDevice genPtr
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.generator_is_cuda genPtr) =
      case unsafePerformIO (cast1 ATen.generator_get_device genPtr) :: Int of
        deviceIndex
          | deviceIndex == fromIntegral (natVal (Proxy @deviceIndex)) -> SDevice SCUDA
          | otherwise ->
            error $
              "The generator should be on CUDA device "
                <> show (natVal (Proxy @deviceIndex))
                <> " but is on device "
                <> show deviceIndex
                <> ". "
                <> gitHubErrorMsg
    | otherwise =
      error $
        "The generator should be on CUDA but is on CPU. "
          <> gitHubErrorMsg

sGetGeneratorDevice ::
  forall device.
  SGetGeneratorDevice device =>
  -- | input
  Generator device ->
  -- | compute device of the input generator
  SDevice device
sGetGeneratorDevice (UnsafeGenerator tvar) = unsafePerformIO . atomically $ do
  state <- readTVar tvar
  case state of
    Left (device, _) -> pure device
    Right genPtr -> pure $ sGetGenPtrDevice genPtr

getGeneratorDeviceType ::
  forall device.
  SGetGeneratorDevice device =>
  -- | input
  Generator device ->
  -- | compute device of the input generator
  DeviceType Int16
getGeneratorDeviceType tensor = forgetIsChecked . fromSing $ sGetGeneratorDevice tensor

getGenPtr ::
  SGetGeneratorDevice device =>
  TVar (Either (SDevice device, Word64) (ForeignPtr ATen.Generator)) ->
  IO (ForeignPtr ATen.Generator)
getGenPtr tvar = do
  state <- atomically $ do
    state <- readTVar tvar
    case state of
      Right genPtr -> do
        let !device = sGetGenPtrDevice genPtr
            !seed = unsafePerformIO $ cast1 ATen.generator_current_seed genPtr
        writeTVar tvar $ Left (device, seed)
        pure (Right genPtr)
      Left _ -> pure state
  case state of
    Right genPtr -> pure genPtr
    Left (device, seed) -> do
      let deviceType = forgetIsChecked . fromSing $ device
      case deviceType of
        CPU -> ATen.newCPUGenerator seed
        CUDA deviceId -> do
          genPtr <- ATen.newCUDAGenerator (fromIntegral deviceId)
          ATen.generator_set_current_seed genPtr seed
          pure genPtr

sCreateWithGenerator ::
  forall m gradient layout device dataType shape generatorDevice.
  (SGetGeneratorDevice generatorDevice, MonadThrow m) =>
  TensorSpec gradient layout device dataType shape ->
  Generator generatorDevice ->
  (ForeignPtr ATen.TensorOptions -> [Dim String Integer] -> ForeignPtr ATen.Generator -> IO (ForeignPtr ATen.Tensor)) ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sCreateWithGenerator TensorSpec {..} (UnsafeGenerator tvar) rawCreateFn =
  unsafeThrowableIO $ do
    genPtr <- getGenPtr tvar
    let TensorOptions opts = tensorOptions tsGradient tsLayout tsDevice tsDataType
        dims = tensorDims tsShape
    tPtr <- rawCreateFn opts dims genPtr
    g <- newTVarIO (Right genPtr)
    return (UnsafeTensor tPtr, UnsafeGenerator g)

sForwardWithGenerator ::
  forall m gradient layout device dataType shape generatorDevice.
  (SGetGeneratorDevice generatorDevice, MonadThrow m) =>
  Tensor gradient layout device dataType shape ->
  Generator generatorDevice ->
  (ForeignPtr ATen.Tensor -> ForeignPtr ATen.Generator -> IO (ForeignPtr ATen.Tensor)) ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sForwardWithGenerator (UnsafeTensor tPtr) (UnsafeGenerator tvar) rawForwardFn =
  unsafeThrowableIO $ do
    genPtr <- getGenPtr tvar
    tPtr' <- rawForwardFn tPtr genPtr
    g <- newTVarIO (Right genPtr)
    return (UnsafeTensor tPtr', UnsafeGenerator g)
