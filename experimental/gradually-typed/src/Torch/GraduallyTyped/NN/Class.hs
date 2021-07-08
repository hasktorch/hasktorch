{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE IncoherentInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C #-}

module Torch.GraduallyTyped.NN.Class where

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (..))
import Control.Monad.State (MonadState (get, put))
import Data.Functor.Const (Const (..))
import Data.Kind (Type)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Typeable (Typeable)
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Debug.Trace (traceShow)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, Nat, natVal, type (+))
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.Layout (SLayout)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..))
import Torch.GraduallyTyped.Shape.Type (SShape)
import Torch.GraduallyTyped.Tensor.Type (Tensor (UnsafeTensor), UncheckedTensor, sCheckedDataType, sCheckedDevice, sCheckedGradient, sCheckedLayout, sCheckedShape)
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe))
import Unsafe.Coerce (unsafeCoerce)

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

instance HasForward (Const () device) input generatorDevice input generatorDevice where
  forward _ = (pure .) . (,)

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

instance
  {-# OVERLAPPABLE #-}
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

class
  HasInitialize
    (model :: Device (DeviceType Nat) -> Type)
    (spec :: Type)
    (device :: Device (DeviceType Nat))
    (generatorDevice :: Device (DeviceType Nat))
    | model -> spec
  where
  initialize ::
    SDevice device ->
    spec ->
    Generator generatorDevice ->
    (model (device <+> generatorDevice), Generator (device <+> generatorDevice))

instance HasInitialize (Const ()) () device generatorDevice where
  initialize _ _ g = (Const (), unsafeCoerce g)

data TDelegate a b c = TDelegate (a c) (b c)

instance
  ( HasInitialize a spec device generatorDevice,
    HasInitialize b spec device (device <+> generatorDevice)
  ) =>
  HasInitialize (TDelegate a b) spec device generatorDevice
  where
  initialize device spec g =
    let (a, g') = initialize device spec g
        (b, g'') = initialize device spec g'
     in (TDelegate a b, g'')

newtype VDelegate n a b = VDelegate (VS.Vector n (a b))

instance
  ( HasInitialize a spec device generatorDevice,
    HasInitialize a spec device (device <+> generatorDevice),
    KnownNat n,
    n' ~ (n + 1)
  ) =>
  HasInitialize (VDelegate n' a) spec device generatorDevice
  where
  initialize device spec g =
    case fromIntegral (natVal (Proxy :: Proxy n) + 1) of
      1 ->
        let (a, g') = initialize device spec g
         in (VDelegate (VGS.Vector (V.singleton a)), g')
      i ->
        let Just (as, (a', g'')) = V.unsnoc $ V.iterateN i (\(_, g') -> initialize device spec g') (initialize device spec g)
         in (VDelegate (VGS.Vector (V.snoc (fst <$> as) a')), g'')

type StateDictKey = String

type StateDict = Map.Map StateDictKey (ForeignPtr ATen.Tensor)

newtype FromStateDictError = FromStateDictKeyNotFoundError {fsdeExpectedKey :: StateDictKey}
  deriving stock (Show, Typeable)

instance Exception FromStateDictError where
  displayException FromStateDictKeyNotFoundError {..} = "`" <> show fsdeExpectedKey <> "` is not in the model's state dictionary."

newtype ToStateDictError = ToStateDictKeyAlreadyInUseError {fsdeTakenKey :: StateDictKey}
  deriving stock (Show, Typeable)

instance Exception ToStateDictError where
  displayException ToStateDictKeyAlreadyInUseError {..} = "`" <> show fsdeTakenKey <> "` is already in the model's state dictionary."

class HasStateDict model input | model -> input where
  fromStateDict :: forall m. (MonadThrow m, MonadState StateDict m) => input -> StateDictKey -> m model
  toStateDict :: forall m. (MonadThrow m, MonadState StateDict m) => StateDictKey -> model -> m ()

instance HasStateDict () () where
  fromStateDict () _ = pure ()
  toStateDict _ _ = pure ()

instance
  HasStateDict
    (Tensor gradient layout device dataType shape)
    (SGradient gradient, SLayout layout, SDevice device, SDataType dataType, SShape shape)
  where
  fromStateDict (gradient, layout, device, dataType, shape) k = do
    traceShow k $ pure ()
    stateDict <- get
    maybe
      (throwM . FromStateDictKeyNotFoundError $ k)
      (\t -> pure (UnsafeTensor t :: UncheckedTensor))
      (Map.lookup k stateDict)
      >>= sCheckedGradient gradient
      >>= sCheckedLayout layout
      >>= sCheckedDevice device
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
  (KnownNat n, HasStateDict a input) =>
  HasStateDict (VS.Vector n a) input
  where
  fromStateDict input k = do
    let i :: Int = fromIntegral (natVal (Proxy :: Proxy n))
        fromStateDict' i' = fromStateDict input (k <> show i' <> ".")
    traverse fromStateDict' . VGS.Vector . V.fromList $ [0 .. i - 1]
  toStateDict k (VGS.Vector v) = do
    let toStateDict' (i', a) = toStateDict (k <> show i' <> ".") a
    mapM_ toStateDict' $ V.zip (V.fromList [0 .. V.length v - 1]) v

stateDictFromPretrained ::
  FilePath ->
  IO StateDict
stateDictFromPretrained filePath = do
  iValue <- Torch.Serialize.pickleLoad filePath
  case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a tensor dictionary."
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."
