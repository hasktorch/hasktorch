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

module Torch.GraduallyTyped.NN.Class where

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (..))
import Control.Monad.State (MonadState (get, put))
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Typeable (Typeable)
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Debug.Trace (traceShow)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, natVal, type (+))
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (SDevice)
import Torch.GraduallyTyped.Layout (SLayout)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..))
import Torch.GraduallyTyped.Shape.Type (SShape)
import Torch.GraduallyTyped.Tensor.Type (Tensor (UnsafeTensor), UncheckedTensor, sCheckedDataType, sCheckedDevice, sCheckedGradient, sCheckedLayout, sCheckedShape)
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe))

class
  HasForward
    model
    input
    generator
    output
    generatorOutput
    | model input generator -> output,
      model input generator -> generatorOutput
  where
  -- | @forward m i g@ for a model @m@, an input @i@, and a generator @g@
  -- returns the tuple @(o, g')@ where @o@ is the output of the model applied to the input
  -- and @g'@ is the updated generator.
  -- @forward m i g@ may throw an exception if the input @i@ or the generator @g@
  -- are not compatible with the model @m@.
  forward :: forall m. MonadThrow m => model -> input -> generator -> m (output, generatorOutput)

instance HasForward () input generator input generator where
  forward _ = (pure .) . (,)

instance
  ( HasForward a input generator output' generatorOutput',
    HasForward b output' generatorOutput' output generatorOutput
  ) =>
  HasForward (a, b) input generator output generatorOutput
  where
  forward (a, b) input g = do
    (output', g') <- forward a input g
    forward b output' g'

instance HasForward (VS.Vector 0 a) input generator input generator where
  forward _ = (pure .) . (,)

instance
  HasForward a input generator output generatorOutput =>
  HasForward (VS.Vector 1 a) input generator output generatorOutput
  where
  forward (VGS.Vector v) input g =
    let Just (a, _) = V.uncons v
     in forward a input g

instance
  {-# OVERLAPPABLE #-}
  ( HasForward a input generator output generatorOutput,
    HasForward a output generatorOutput output generatorOutput
  ) =>
  HasForward (VS.Vector n a) input generator output generatorOutput
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
  HasInitialize model input generator generatorOutput
    | model -> input,
      model generator -> generatorOutput
  where
  initialize :: input -> generator -> (model, generatorOutput)

instance HasInitialize () () generator generator where
  initialize _ g = ((), g)

instance
  ( HasInitialize a input generator generatorOutput',
    HasInitialize b input generatorOutput' generatorOutput
  ) =>
  HasInitialize (a, b) input generator generatorOutput
  where
  initialize input g =
    let (a, g') = initialize @a input g
        (b, g'') = initialize @b input g'
     in ((a, b), g'')

instance
  ( HasInitialize a input generator generatorOutput,
    HasInitialize a input generatorOutput generatorOutput,
    KnownNat n,
    n' ~ (n + 1)
  ) =>
  HasInitialize (VS.Vector n' a) input generator generatorOutput
  where
  initialize input g =
    case fromIntegral (natVal (Proxy :: Proxy n) + 1) of
      1 ->
        let (a, g') = initialize @a input g
         in (VGS.Vector (V.singleton a), g')
      i ->
        let Just (as, (a', g'')) = V.unsnoc $ V.iterateN i (\(_, g') -> initialize @a input g') (initialize @a input g)
         in (VGS.Vector (V.snoc (fst <$> as) a'), g'')

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
