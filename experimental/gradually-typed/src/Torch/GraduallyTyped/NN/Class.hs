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

module Torch.GraduallyTyped.NN.Class where

-- import Control.Monad.State.Strict (MonadState (state), runState)
-- import Torch.GraduallyTyped.Prelude (Contains, ErrorMessage (Text), Fst, If, Proxy (..), Snd, Type, TypeError)
-- import Torch.GraduallyTyped.Random (Generator)
-- import Torch.GraduallyTyped.Device (Device (UncheckedDevice), DeviceType)
-- import Generics.SOP (Code, I, SOP(..), Generic, NS(..), NP)
-- import GHC.Base (coerce, Any)

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (..))
import Control.Monad.State (MonadState (get, put))
import Data.Kind (Type)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (..))
import Data.Typeable (Typeable)
import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (ErrorMessage (..), KnownNat, TypeError, natVal, type (+))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), SLayout)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Type (KnownShape, SShape, Shape (..))
import Torch.GraduallyTyped.Tensor.Type (DataTypeError, DeviceError, LayoutError, ShapeError, Tensor (UnsafeTensor), UncheckedTensor, checkedDataType, checkedDevice, checkedLayout, checkedShape, sCheckedDataType, sCheckedDevice, sCheckedGradient, sCheckedLayout, sCheckedShape)
import qualified Torch.Internal.Type as ATen (Tensor)

-- class Foo a i o | a i -> o where
--   foo :: a -> i -> o

-- instance
--   ( Foo a i o,
--     Foo a o o
--   ) =>
--   Foo (a, a) i o
--   where
--   foo (a, a') i = foo a' $ foo a i

-- data A = A

-- instance TypeError ('Text "no Bool!") => Foo A Bool () where
--   foo _ _ = ()

-- instance TypeError ('Text "no ()!") => Foo A () () where
--   foo _ _ = ()

-- bar = foo (A, A) True

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
  -- | @forward m i g@ for a model @m@, an input @o@, and a generator @g@
  -- returns the tuple @(o, g')@ where @o@ is the output of the model applied to the input
  -- and @g'@ is the updated generator.
  forward :: model -> input -> generator -> (output, generatorOutput)

instance HasForward () input generator input generator where
  forward _ = (,)

instance
  ( HasForward a input generator output' generatorOutput',
    HasForward b output' generatorOutput' output generatorOutput
  ) =>
  HasForward (a, b) input generator output generatorOutput
  where
  forward (a, b) input g =
    let (output', g') = forward a input g
     in forward b output' g'

instance HasForward (VS.Vector 0 a) input generator input generator where
  forward _ = (,)

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
     in V.foldl (\(output', g') a' -> forward a' output' g') (forward a input g) as

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

instance HasInitialize (VS.Vector 0 a) () generator generator where
  initialize _ g = (VGS.Vector V.empty, g)

instance
  HasInitialize a input generator generatorOutput =>
  HasInitialize (VS.Vector 1 a) input generator generatorOutput
  where
  initialize input g = let (a, g') = initialize @a input g in (VGS.Vector (V.singleton a), g')

instance
  {-# OVERLAPPABLE #-}
  ( HasInitialize a input generator generatorOutput,
    HasInitialize a input generatorOutput generatorOutput,
    KnownNat n
  ) =>
  HasInitialize (VS.Vector n a) input generator generatorOutput
  where
  initialize input g =
    let i = fromIntegral (natVal (Proxy :: Proxy n))
        Just (as, (a', g'')) = V.unsnoc $ V.iterateN i (\(_, g') -> initialize @a input g') (initialize @a input g)
     in (VGS.Vector (V.snoc (fst <$> as) a'), g'')

testInitializeVector ::
  forall n generator.
  KnownNat n =>
  HasInitialize (VS.Vector n ()) () generator generator =>
  (generator -> (VS.Vector n (), generator))
testInitializeVector = initialize @(VS.Vector n ()) ()

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

class
  HasStateDict
    model
    input
    | model -> input
  where
  fromStateDict ::
    forall e m.
    ( MonadThrow m,
      MonadState StateDict m
    ) =>
    input ->
    StateDictKey ->
    m model
  toStateDict ::
    forall e m.
    ( MonadThrow m,
      MonadState StateDict m
    ) =>
    StateDictKey ->
    model ->
    m ()

instance
  HasStateDict
    (Tensor gradient layout device dataType shape)
    (SGradient gradient, SLayout layout, SDevice device, SDataType dataType, SShape shape)
  where
  fromStateDict (gradient, layout, device, dataType, shape) k = do
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

-- class GHasForward model input where
--   type GOutput model input
--   gForward :: (Generic model, Generic input, Code model ~ Code input) => model -> input -> GOutput model input
--   gForward model input = gForwardSS (from model) (from input)

-- class GHasForwardSS modelss inputss where
--   type GOutputSS modelss inputss
--   gForwardSS :: forall models inputs . (GHasForwardPP models inputs) => SOP I modelss -> SOP I inputss -> GOutputSS modelss inputss
--   gForwardSS (SOP (Z (models :: NP I models))) (SOP (Z (inputs :: NP I inputs))) = gForwardPP @models @inputs models inputs

-- class GHasForwardPP models inputs where
--   type GOutputPP models inputs
--   gForwardPP :: () => NP I models -> NP I inputs -> GOutputPP models inputs

-- data ModelRandomness = Deterministic | Stochastic

-- type family ModelRandomnessR (output :: Type) :: (ModelRandomness, Type) where
--   ModelRandomnessR (Generator device -> (output, Generator device)) =
--     If
--       (Contains output Generator)
--       (TypeError (Text "The random generator appears in a wrong position in the output type."))
--       '( 'Stochastic, output)
--   ModelRandomnessR output =
--     If
--       (Contains output Generator)
--       (TypeError (Text "The random generator appears in a wrong position in the output type."))
--       '( 'Deterministic, output)

-- class
--   HasForwardProduct
--     (modelARandomness :: ModelRandomness)
--     outputA
--     (modelBRandomness :: ModelRandomness)
--     outputB
--     modelA
--     inputA
--     modelB
--     inputB
--   where
--   type
--     ForwardOutputProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB ::
--       Type
--   forwardProduct ::
--     Proxy modelARandomness ->
--     Proxy outputA ->
--     Proxy modelBRandomness ->
--     Proxy outputB ->
--     (modelA, modelB) ->
--     (inputA, inputB) ->
--     ForwardOutputProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB

-- instance
--   ( HasForward modelA inputA,
--     ForwardOutput modelA inputA 'UncheckedDevice ~ (Generator 'UncheckedDevice -> (outputA, Generator 'UncheckedDevice)),
--     HasForward modelB inputB,
--     ForwardOutput modelB inputB ~ (Generator 'UncheckedDevice -> (outputB, Generator 'UncheckedDevice))
--   ) =>
--   HasForwardProduct 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB
--   where
--   type
--     ForwardOutputProduct 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB =
--       Generator 'UncheckedDevice -> ((outputA, outputB), Generator 'UncheckedDevice)
--   forwardProduct _ _ _ _ (modelA, modelB) (inputA, inputB) =
--     runState $ do
--       outputA <- state (forward modelA inputA)
--       outputB <- state (forward modelB inputB)
--       return (outputA, outputB)

-- instance
--   ( '(modelARandomness, outputA) ~ ModelRandomnessR (ForwardOutput modelA inputA),
--     '(modelBRandomness, outputB) ~ ModelRandomnessR (ForwardOutput modelB inputB),
--     HasForwardProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB
--   ) =>
--   HasForward (modelA, modelB) (inputA, inputB)
--   where
--   type
--     ForwardOutput (modelA, modelB) (inputA, inputB) =
--       ForwardOutputProduct
--         (Fst (ModelRandomnessR (ForwardOutput modelA inputA)))
--         (Snd (ModelRandomnessR (ForwardOutput modelA inputA)))
--         (Fst (ModelRandomnessR (ForwardOutput modelB inputB)))
--         (Snd (ModelRandomnessR (ForwardOutput modelB inputB)))
--         modelA
--         inputA
--         modelB
--         inputB
--   forward =
--     forwardProduct
--       (Proxy :: Proxy modelARandomness)
--       (Proxy :: Proxy outputA)
--       (Proxy :: Proxy modelBRandomness)
--       (Proxy :: Proxy outputB)

-- class
--   HasForwardSum
--     (modelARandomness :: ModelRandomness)
--     outputA
--     (modelBRandomness :: ModelRandomness)
--     outputB
--     modelA
--     inputA
--     modelB
--     inputB
--   where
--   type
--     ForwardOutputSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB ::
--       Type
--   forwardSum ::
--     Proxy modelARandomness ->
--     Proxy outputA ->
--     Proxy modelBRandomness ->
--     Proxy outputB ->
--     Either modelA modelB ->
--     Either inputA inputB ->
--     ForwardOutputSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB

-- instance
--   ( HasForward modelA inputA,
--     ForwardOutput modelA inputA ~ (Generator 'UncheckedDevice -> (outputA, Generator 'UncheckedDevice)),
--     HasForward modelB inputB,
--     ForwardOutput modelB inputB ~ (Generator 'UncheckedDevice -> (outputB, Generator 'UncheckedDevice))
--   ) =>
--   HasForwardSum 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB
--   where
--   type
--     ForwardOutputSum 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB =
--       Generator 'UncheckedDevice -> (Maybe (Either outputA outputB), Generator 'UncheckedDevice)
--   forwardSum _ _ _ _ (Left modelA) (Left inputA) =
--     runState $ Just . Left <$> (state $ forward modelA inputA)
--   forwardSum _ _ _ _ (Right modelB) (Right inputB) =
--     runState $ Just . Right <$> (state $ forward modelB inputB)
--   forwardSum _ _ _ _ _ _ = runState . pure $ Nothing

-- instance
--   ( '(modelARandomness, outputA) ~ ModelRandomnessR (ForwardOutput modelA inputA),
--     '(modelBRandomness, outputB) ~ ModelRandomnessR (ForwardOutput modelB inputB),
--     HasForwardSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB
--   ) =>
--   HasForward (Either modelA modelB) (Either inputA inputB)
--   where
--   type
--     ForwardOutput (Either modelA modelB) (Either inputA inputB) =
--       ForwardOutputSum
--         (Fst (ModelRandomnessR (ForwardOutput modelA inputA)))
--         (Snd (ModelRandomnessR (ForwardOutput modelA inputA)))
--         (Fst (ModelRandomnessR (ForwardOutput modelB inputB)))
--         (Snd (ModelRandomnessR (ForwardOutput modelB inputB)))
--         modelA
--         inputA
--         modelB
--         inputB
--   forward =
--     forwardSum
--       (Proxy :: Proxy modelARandomness)
--       (Proxy :: Proxy outputA)
--       (Proxy :: Proxy modelBRandomness)
--       (Proxy :: Proxy outputB)

-- data ModelA = ModelA

-- data InputA = InputA

-- data OutputA = OutputA

-- instance HasForward ModelA InputA where
--   type ForwardOutput ModelA InputA = (Generator 'UncheckedDevice -> (OutputA, Generator 'UncheckedDevice))
--   forward _ _ g = (OutputA, g)

-- data ModelB = ModelB

-- data InputB = InputB

-- data OutputB = OutputB

-- instance HasForward ModelB InputB where
--   type ForwardOutput ModelB InputB = (Generator 'UncheckedDevice -> (OutputB, Generator 'UncheckedDevice))
--   forward _ _ g = (OutputB, g)

-- test :: Generator 'UncheckedDevice -> ((OutputA, OutputB), Generator 'UncheckedDevice)
-- test = forward (ModelA, ModelB) (InputA, InputB)

-- test' :: Generator 'UncheckedDevice -> (Maybe (Either OutputA OutputB), Generator 'UncheckedDevice)
-- test' = forward @(Either ModelA ModelB) @(Either InputA InputB) (Left ModelA) (Right InputB)
