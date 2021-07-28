{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C #-}

module Torch.GraduallyTyped.NN.Class where

import Control.Exception (Exception (..))
import Control.Monad (void)
import Control.Monad.Catch (MonadThrow (..))
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.State (MonadState (get, put))
import Data.Bifunctor (Bifunctor (bimap))
import Data.Kind (Type)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Singletons.TypeLits (SNat (..))
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Typeable (Typeable)
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, natVal, type (+))
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Tensor.Type (SSetDevice (sSetDevice), SSetGradient (..), Tensor (..), TensorSpec (..), UncheckedTensor, sCheckedDataType, sCheckedLayout, sCheckedShape)
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad, pickleSave)
import qualified Torch.Tensor (Tensor (Unsafe))

data NamedModel model = NamedModel Text model

class
  HasForward
    model
    input
    generatorDevice
    output
    generatorOutputDevice
    | model input generatorDevice -> output,
      model input generatorDevice -> generatorOutputDevice
  where
  -- | @forward m i g@ for a model @m@, an input @i@, and a generator @g@
  -- returns the tuple @(o, g')@ where @o@ is the output of the model applied to the input
  -- and @g'@ is the updated generator.
  -- @forward m i g@ may throw an exception if the input @i@ or the generator @g@
  -- are not compatible with the model @m@.
  forward :: forall m. MonadThrow m => model -> input -> Generator generatorDevice -> m (output, Generator generatorOutputDevice)

instance HasForward () input generatorDevice input generatorDevice where
  forward _ = (pure .) . (,)

instance
  HasForward model input generatorDevice output generatorOutputDevice =>
  HasForward (NamedModel model) input generatorDevice output generatorOutputDevice
  where
  forward (NamedModel _ model) = forward model

instance
  ( HasForward a input generatorDevice output' generatorOutputDevice',
    HasForward b output' generatorOutputDevice' output generatorOutputDevice
  ) =>
  HasForward (a, b) input generatorDevice output generatorOutputDevice
  where
  forward (a, b) input g = do
    (output', g') <- forward a input g
    forward b output' g'

instance HasForward (VS.Vector 0 a) input generatorDevice input generatorDevice where
  forward _ = (pure .) . (,)

instance
  HasForward a input generatorDevice output generatorOutputDevice =>
  HasForward (VS.Vector 1 a) input generatorDevice output generatorOutputDevice
  where
  forward (VGS.Vector v) input g =
    let Just (a, _) = V.uncons v
     in forward a input g

instance-- {-# OVERLAPPABLE #-}

  ( HasForward a input generatorDevice output generatorOutputDevice,
    HasForward a output generatorOutputDevice output generatorOutputDevice
  ) =>
  HasForward (VS.Vector n a) input generatorDevice output generatorOutputDevice
  where
  forward (VGS.Vector v) input g =
    let Just (a, as) = V.uncons v
     in V.foldl
          ( \agg a' -> do
              (output', g') <- agg
              forward a' output' g'
          )
          (forward a input g)
          as

type family ModelSpec model = (spec :: Type) | spec -> model

class
  HasInitialize
    (model :: Type)
    (generatorDevice :: Device (DeviceType Nat))
    (output :: Type)
    (generatorOutputDevice :: Device (DeviceType Nat))
    | model generatorDevice -> output,
      model generatorDevice -> generatorOutputDevice
  where
  initialize ::
    forall m.
    MonadThrow m =>
    ModelSpec model ->
    Generator generatorDevice ->
    m (output, Generator generatorOutputDevice)

type instance ModelSpec () = ()

instance HasInitialize () generatorDevice () generatorDevice where
  initialize () g = pure ((), g)

type instance ModelSpec (NamedModel model) = NamedModel (ModelSpec model)

instance
  HasInitialize model generatorDevice output generatorOutputDevice =>
  HasInitialize (NamedModel model) generatorDevice (NamedModel output) generatorOutputDevice
  where
  initialize (NamedModel modelName modelSpec) g = do
    (model, g') <- initialize modelSpec g
    pure (NamedModel modelName model, g')

type instance ModelSpec (a, b) = (ModelSpec a, ModelSpec b)

instance
  ( HasInitialize a generatorDevice aOutput aGeneratorOutputDevice,
    HasInitialize b aGeneratorOutputDevice bOutput generatorOutputDevice
  ) =>
  HasInitialize (a, b) generatorDevice (aOutput, bOutput) generatorOutputDevice
  where
  initialize (aSpec, bSpec) g = do
    (a, g') <- initialize aSpec g
    (b, g'') <- initialize bSpec g'
    pure ((a, b), g'')

data VectorSpec (n :: Nat) (a :: Type) where
  VectorSpec ::
    forall n a.
    SNat n ->
    VS.Vector n (ModelSpec a) ->
    VectorSpec n a

type instance ModelSpec (VS.Vector n a) = VectorSpec n a

instance
  ( HasInitialize a generatorDevice output generatorOutputDevice,
    HasInitialize a generatorOutputDevice output generatorOutputDevice,
    n' ~ (n + 1)
  ) =>
  HasInitialize (VS.Vector n' a) generatorDevice (VS.Vector n' output) generatorOutputDevice
  where
  initialize (VectorSpec SNat (VGS.Vector specs)) g = do
    let Just (spec, specs') = V.uncons specs
    (a, g') <- initialize spec g
    (as, g'''') <-
      V.foldl
        ( \agg spec' -> do
            (acc, g'') <- agg
            (a', g''') <- initialize spec' g''
            pure (V.snoc acc a', g''')
        )
        (pure (V.singleton a, g'))
        specs'
    pure (VGS.Vector as, g'''')

type StateDictKey = Text

type StateDict = Map.Map StateDictKey (ForeignPtr ATen.Tensor)

newtype FromStateDictError = FromStateDictKeyNotFoundError {fsdeExpectedKey :: StateDictKey}
  deriving stock (Show, Typeable)

instance Exception FromStateDictError where
  displayException FromStateDictKeyNotFoundError {..} = "`" <> show fsdeExpectedKey <> "` is not in the model's state dictionary."

newtype ToStateDictError = ToStateDictKeyAlreadyInUseError {fsdeTakenKey :: StateDictKey}
  deriving stock (Show, Typeable)

instance Exception ToStateDictError where
  displayException ToStateDictKeyAlreadyInUseError {..} = "`" <> show fsdeTakenKey <> "` is already in the model's state dictionary."

class HasStateDict model where
  fromStateDict ::
    forall m.
    (MonadIO m, MonadThrow m, MonadState StateDict m) =>
    ModelSpec model ->
    StateDictKey ->
    m model
  toStateDict ::
    forall m.
    (MonadThrow m, MonadState StateDict m) =>
    StateDictKey ->
    model ->
    m ()

instance HasStateDict () where
  fromStateDict () _ = pure ()
  toStateDict _ () = pure ()

instance HasStateDict model => HasStateDict (NamedModel model) where
  fromStateDict (NamedModel modelName modelSpec) key =
    NamedModel modelName <$> fromStateDict modelSpec (key <> modelName)
  toStateDict key (NamedModel modelName model) =
    toStateDict (key <> modelName) model

instance (HasStateDict a, HasStateDict b) => HasStateDict (a, b) where
  fromStateDict (aSpec, bSpec) k =
    (,)
      <$> fromStateDict aSpec (k <> "0.")
      <*> fromStateDict bSpec (k <> "1.")
  toStateDict k (a, b) = do
    void $ toStateDict (k <> "0.") a
    void $ toStateDict (k <> "1.") b

type instance ModelSpec (Tensor gradient layout device dataType shape) = TensorSpec gradient layout device dataType shape

instance
  (SSetGradient gradient, SSetDevice device) =>
  HasStateDict
    (Tensor gradient layout device dataType shape)
  where
  fromStateDict (TensorSpec gradient layout device dataType shape) k = do
    stateDict <- get
    maybe
      (throwM . FromStateDictKeyNotFoundError $ k)
      (\t -> pure (UnsafeTensor t :: UncheckedTensor))
      (Map.lookup k stateDict)
      >>= liftIO . sSetGradient gradient
      >>= sCheckedLayout layout
      >>= sSetDevice device
      >>= sCheckedDataType dataType
      >>= sCheckedShape shape
  toStateDict k (UnsafeTensor t) = do
    stateDict <- get
    stateDict' <-
      maybe
        (pure $ Map.insert k t stateDict)
        (\_ -> throwM . ToStateDictKeyAlreadyInUseError $ k)
        (Map.lookup k stateDict)
    put stateDict'

instance
  HasStateDict a =>
  HasStateDict (VS.Vector n a)
  where
  fromStateDict (VectorSpec SNat specs) k = do
    let i :: Int = fromIntegral (natVal (Proxy :: Proxy n))
        fromStateDict' (spec, i') = fromStateDict spec (k <> Text.pack (show i') <> ".")
    traverse fromStateDict' $ VS.zip specs (VGS.Vector $ V.fromList [0 .. i - 1])
  toStateDict k (VGS.Vector v) = do
    let toStateDict' (i', a) = toStateDict (k <> Text.pack (show i') <> ".") a
    mapM_ toStateDict' $ V.zip (V.fromList [0 .. V.length v - 1]) v

-- | Load a state dictionary from a TorchScript file.
stateDictFromFile ::
  FilePath ->
  IO StateDict
stateDictFromFile filePath = do
  iValue <- Torch.Serialize.pickleLoad filePath
  case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a tensor dictionary."
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((Text.pack s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."

-- | Save a state dictionary to a TorchScript file.
stateDictToFile ::
  StateDict ->
  FilePath ->
  IO ()
stateDictToFile stateDict filePath = do
  let iValue =
        Torch.Script.IVGenericDict $
          bimap
            (Torch.Script.IVString . Text.unpack)
            (Torch.Script.IVTensor . Torch.Tensor.Unsafe)
            <$> Map.toList stateDict
  Torch.Serialize.pickleSave iValue filePath