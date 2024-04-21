{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Class where

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (..))
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (MonadState (get, put))
import Data.Bifunctor (Bifunctor (bimap))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Constraint, Type)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Typeable (Typeable)
import qualified Data.Vector as V hiding (uncons)
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Foreign.ForeignPtr (ForeignPtr)
import GHC.Generics (Generic (..), K1 (..), M1 (..), U1 (..), (:*:) (..))
import GHC.TypeLits -- (Nat, natVal, type (+), SNat(..), SNat)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import qualified Torch.GraduallyTyped.Internal.Vector as V
-- import Torch.GraduallyTyped.Prelude.TypeLits (SNat (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape.Type (SDim)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), TensorSpec (..), UncheckedTensor, sCheckedDataType, sCheckedLayout, sCheckedShape, sSetDevice, sSetGradient)
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad, pickleSave)
import qualified Torch.Tensor (Tensor (Unsafe))
-- import GHC.TypeNats
--import GHC.TypeLits


type NamedModel :: Type -> Type
data NamedModel model = NamedModel Text model
  deriving stock (Eq, Ord, Show, Generic)

pattern (::>) :: Text -> model -> NamedModel model
pattern (::>) name model = NamedModel name model

type HasForward ::
  Type ->
  Type ->
  Device (DeviceType Nat) ->
  Type ->
  Device (DeviceType Nat) ->
  Constraint
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
  forward ::
    forall m.
    MonadThrow m =>
    -- | model
    model ->
    -- | model input, typically a tensor or a tuple of tensors
    input ->
    -- | random generator
    Generator generatorDevice ->
    -- | output of the model with an updated generator
    m (output, Generator generatorOutputDevice)
  default forward ::
    forall m.
    ( MonadThrow m,
      Generic model,
      GHasForward (Rep model) input generatorDevice output generatorOutputDevice
    ) =>
    model ->
    input ->
    Generator generatorDevice ->
    m (output, Generator generatorOutputDevice)
  forward model = gForward (from model)

instance HasForward () input generatorDevice input generatorDevice where
  forward _ = (pure .) . (,)

instance
  HasForward model input generatorDevice output generatorOutputDevice =>
  HasForward (NamedModel model) input generatorDevice output generatorOutputDevice
  where
  forward (NamedModel _ model) = forward model

type GHasForward ::
  (Type -> Type) ->
  Type ->
  Device (DeviceType Nat) ->
  Type ->
  Device (DeviceType Nat) ->
  Constraint
class
  GHasForward
    gModel
    input
    generatorDevice
    output
    generatorOutputDevice
    | gModel input generatorDevice -> output,
      gModel input generatorDevice -> generatorOutputDevice
  where
  gForward ::
    forall m c.
    MonadThrow m =>
    gModel c ->
    input ->
    Generator generatorDevice ->
    m (output, Generator generatorOutputDevice)

instance
  ( GHasForward
      gModelA
      inputA
      generatorDevice
      outputA
      generatorOutputADevice,
    GHasForward
      gModelB
      outputA
      generatorOutputADevice
      outputB
      generatorOutputDevice
  ) =>
  GHasForward
    (gModelA :*: gModelB)
    inputA
    generatorDevice
    outputB
    generatorOutputDevice
  where
  gForward (gModelA :*: gModelB) input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . gForward gModelA
        >>>= IxStateT . gForward gModelB

instance
  GHasForward gModel input generatorDevice output generatorOutputDevice =>
  GHasForward
    (M1 i t gModel)
    input
    generatorDevice
    output
    generatorOutputDevice
  where
  gForward (M1 gModel) = gForward gModel

instance
  HasForward model input generatorDevice output generatorOutputDevice =>
  GHasForward
    (K1 i model)
    input
    generatorDevice
    output
    generatorOutputDevice
  where
  gForward (K1 model) = forward model

instance GHasForward U1 input generatorDevice input generatorDevice where
  gForward U1 input g = pure (input, g)

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice output generatorOutputDevice
  ) =>
  HasForward (a, b) input generatorDevice output generatorOutputDevice

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice outputB generatorOutputBDevice,
    HasForward c outputB generatorOutputBDevice output generatorOutputDevice
  ) =>
  HasForward (a, b, c) input generatorDevice output generatorOutputDevice

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice outputB generatorOutputBDevice,
    HasForward c outputB generatorOutputBDevice outputC generatorOutputCDevice,
    HasForward d outputC generatorOutputCDevice output generatorOutputDevice
  ) =>
  HasForward (a, b, c, d) input generatorDevice output generatorOutputDevice

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice outputB generatorOutputBDevice,
    HasForward c outputB generatorOutputBDevice outputC generatorOutputCDevice,
    HasForward d outputC generatorOutputCDevice outputD generatorOutputDDevice,
    HasForward e outputD generatorOutputDDevice output generatorOutputDevice
  ) =>
  HasForward (a, b, c, d, e) input generatorDevice output generatorOutputDevice

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice outputB generatorOutputBDevice,
    HasForward c outputB generatorOutputBDevice outputC generatorOutputCDevice,
    HasForward d outputC generatorOutputCDevice outputD generatorOutputDDevice,
    HasForward e outputD generatorOutputDDevice outputE generatorOutputEDevice,
    HasForward f outputE generatorOutputEDevice output generatorOutputDevice
  ) =>
  HasForward (a, b, c, d, e, f) input generatorDevice output generatorOutputDevice

instance
  ( HasForward a input generatorDevice outputA generatorOutputADevice,
    HasForward b outputA generatorOutputADevice outputB generatorOutputBDevice,
    HasForward c outputB generatorOutputBDevice outputC generatorOutputCDevice,
    HasForward d outputC generatorOutputCDevice outputD generatorOutputDDevice,
    HasForward e outputD generatorOutputDDevice outputE generatorOutputEDevice,
    HasForward f outputE generatorOutputEDevice outputF generatorOutputFDevice,
    HasForward g outputF generatorOutputFDevice output generatorOutputDevice
  ) =>
  HasForward (a, b, c, d, e, f, g) input generatorDevice output generatorOutputDevice

type Wrap :: Type -> Type
newtype Wrap a = Wrap a
  deriving (Eq, Ord, Show, Generic)

type instance ModelSpec (Wrap a) = Wrap (ModelSpec a)

instance
  HasInitialize a generatorDevice a' generatorOutputDevice =>
  HasInitialize (Wrap a) generatorDevice (Wrap a') generatorOutputDevice

instance
  HasStateDict a =>
  HasStateDict (Wrap a)

instance
  HasForward a input generatorDevice output generatorOutputDevice =>
  HasForward (Wrap a) input generatorDevice output generatorOutputDevice

type ListToTuple :: [Type] -> Type
type family ListToTuple xs = tuple | tuple -> xs where
  ListToTuple '[] = ()
  ListToTuple '[a] = Wrap a
  ListToTuple '[a, b] = (a, b)
  ListToTuple '[a, b, c] = (a, b, c)
  ListToTuple '[a, b, c, d] = (a, b, c, d)
  ListToTuple '[a, b, c, d, e] = (a, b, c, d, e)
  ListToTuple '[a, b, c, d, e, f] = (a, b, c, d, e, f)
  ListToTuple '[a, b, c, d, e, f, g] = (a, b, c, d, e, f, g)

type ModelStack :: [Type] -> Type
newtype ModelStack models = ModelStack (ListToTuple models)
  deriving stock (Generic)

type instance ModelSpec (ModelStack '[]) = ModelStack '[]

type instance ModelSpec (ModelStack '[a]) = ModelStack '[ModelSpec a]

type instance ModelSpec (ModelStack '[a, b]) = ModelStack '[ModelSpec a, ModelSpec b]

type instance ModelSpec (ModelStack '[a, b, c]) = ModelStack '[ModelSpec a, ModelSpec b, ModelSpec c]

type instance ModelSpec (ModelStack '[a, b, c, d]) = ModelStack '[ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d]

type instance ModelSpec (ModelStack '[a, b, c, d, e]) = ModelStack '[ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e]

type instance ModelSpec (ModelStack '[a, b, c, d, e, f]) = ModelStack '[ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e, ModelSpec f]

type instance ModelSpec (ModelStack '[a, b, c, d, e, f, g]) = ModelStack '[ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e, ModelSpec f, ModelSpec g]

instance
  HasForward (ListToTuple models) input generatorDevice output generatorOutputDevice =>
  HasForward (ModelStack models) input generatorDevice output generatorOutputDevice
  where
  forward (ModelStack models) = forward models

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

type ModelSpec :: Type -> Type
type family ModelSpec model = spec | spec -> model

type HasInitialize ::
  Type ->
  Device (DeviceType Nat) ->
  Type ->
  Device (DeviceType Nat) ->
  Constraint
class
  HasInitialize
    model
    generatorDevice
    output
    generatorOutputDevice
    | model generatorDevice -> output,
      model generatorDevice -> generatorOutputDevice
  where
  initialize ::
    forall m.
    MonadThrow m =>
    ModelSpec model ->
    Generator generatorDevice ->
    m (output, Generator generatorOutputDevice)
  default initialize ::
    forall m.
    ( MonadThrow m,
      Generic (ModelSpec model),
      Generic output,
      GHasInitialize (Rep (ModelSpec model)) generatorDevice (Rep output) generatorOutputDevice
    ) =>
    ModelSpec model ->
    Generator generatorDevice ->
    m (output, Generator generatorOutputDevice)
  initialize modelSpec =
    runIxStateT $
      IxStateT (gInitialize (from modelSpec))
        >>>= ireturn . to

type GHasInitialize ::
  (Type -> Type) ->
  Device (DeviceType Nat) ->
  (Type -> Type) ->
  Device (DeviceType Nat) ->
  Constraint
class
  GHasInitialize
    gModelSpec
    generatorDevice
    gOutput
    generatorOutputDevice
    | gModelSpec generatorDevice -> gOutput,
      gModelSpec generatorDevice -> generatorOutputDevice
  where
  gInitialize ::
    forall m c.
    MonadThrow m =>
    gModelSpec c ->
    Generator generatorDevice ->
    m (gOutput c, Generator generatorOutputDevice)

instance
  ( GHasInitialize gModelSpecA generatorDevice gOutputA generatorOutputADevice,
    GHasInitialize gModelSpecB generatorOutputADevice gOutputB generatorOutputDevice
  ) =>
  GHasInitialize
    (gModelSpecA :*: gModelSpecB)
    generatorDevice
    (gOutputA :*: gOutputB)
    generatorOutputDevice
  where
  gInitialize (gModelSpecA :*: gModelSpecB) =
    runIxStateT $
      (:*:) <<$>> IxStateT (gInitialize gModelSpecA) <<*>> IxStateT (gInitialize gModelSpecB)

instance
  GHasInitialize gModelSpec generatorDevice gOutput generatorOutputDevice =>
  GHasInitialize
    (M1 i t gModelSpec)
    generatorDevice
    (M1 i t gOutput)
    generatorOutputDevice
  where
  gInitialize (M1 gModelSpec) = runIxStateT $ M1 <<$>> IxStateT (gInitialize gModelSpec)

instance
  ( HasInitialize model generatorDevice output generatorOutputDevice,
    ModelSpec model ~ modelSpec
  ) =>
  GHasInitialize
    (K1 i modelSpec)
    generatorDevice
    (K1 i output)
    generatorOutputDevice
  where
  gInitialize (K1 modelSpec) = runIxStateT $ K1 <<$>> IxStateT (initialize @model modelSpec)

instance GHasInitialize U1 generatorDevice U1 generatorDevice where
  gInitialize U1 g = pure (U1, g)

type instance ModelSpec (SDim dim) = SDim dim

instance HasInitialize (SDim dim) generatorDevice (SDim dim) generatorDevice where
  initialize dim g = pure (dim, g)

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

type instance ModelSpec (a, b, c) = (ModelSpec a, ModelSpec b, ModelSpec c)

type instance ModelSpec (a, b, c, d) = (ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d)

type instance ModelSpec (a, b, c, d, e) = (ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e)

type instance ModelSpec (a, b, c, d, e, f) = (ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e, ModelSpec f)

type instance ModelSpec (a, b, c, d, e, f, g) = (ModelSpec a, ModelSpec b, ModelSpec c, ModelSpec d, ModelSpec e, ModelSpec f, ModelSpec g)

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputDevice
  ) =>
  HasInitialize (a, b) generatorDevice (outputA, outputB) generatorOutputDevice

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputBDevice,
    HasInitialize c generatorOutputBDevice outputC generatorOutputDevice
  ) =>
  HasInitialize (a, b, c) generatorDevice (outputA, outputB, outputC) generatorOutputDevice

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputBDevice,
    HasInitialize c generatorOutputBDevice outputC generatorOutputCDevice,
    HasInitialize d generatorOutputCDevice outputD generatorOutputDevice
  ) =>
  HasInitialize (a, b, c, d) generatorDevice (outputA, outputB, outputC, outputD) generatorOutputDevice

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputBDevice,
    HasInitialize c generatorOutputBDevice outputC generatorOutputCDevice,
    HasInitialize d generatorOutputCDevice outputD generatorOutputDDevice,
    HasInitialize e generatorOutputDDevice outputE generatorOutputDevice
  ) =>
  HasInitialize (a, b, c, d, e) generatorDevice (outputA, outputB, outputC, outputD, outputE) generatorOutputDevice

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputBDevice,
    HasInitialize c generatorOutputBDevice outputC generatorOutputCDevice,
    HasInitialize d generatorOutputCDevice outputD generatorOutputDDevice,
    HasInitialize e generatorOutputDDevice outputE generatorOutputEDevice,
    HasInitialize f generatorOutputEDevice outputF generatorOutputDevice
  ) =>
  HasInitialize (a, b, c, d, e, f) generatorDevice (outputA, outputB, outputC, outputD, outputE, outputF) generatorOutputDevice

instance
  ( HasInitialize a generatorDevice outputA generatorOutputADevice,
    HasInitialize b generatorOutputADevice outputB generatorOutputBDevice,
    HasInitialize c generatorOutputBDevice outputC generatorOutputCDevice,
    HasInitialize d generatorOutputCDevice outputD generatorOutputDDevice,
    HasInitialize e generatorOutputDDevice outputE generatorOutputEDevice,
    HasInitialize f generatorOutputEDevice outputF generatorOutputFDevice,
    HasInitialize g generatorOutputFDevice outputG generatorOutputDevice
  ) =>
  HasInitialize (a, b, c, d, e, f, g) generatorDevice (outputA, outputB, outputC, outputD, outputE, outputF, outputG) generatorOutputDevice

instance HasInitialize (ModelStack '[]) generatorDevice (ModelStack '[]) generatorDevice

instance
  HasInitialize a generatorDevice a' generatorOutputDevice =>
  HasInitialize (ModelStack '[a]) generatorDevice (ModelStack '[a']) generatorOutputDevice

instance
  HasInitialize (a, b) generatorDevice (a', b') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b]) generatorDevice (ModelStack '[a', b']) generatorOutputDevice

instance
  HasInitialize (a, b, c) generatorDevice (a', b', c') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b, c]) generatorDevice (ModelStack '[a', b', c']) generatorOutputDevice

instance
  HasInitialize (a, b, c, d) generatorDevice (a', b', c', d') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b, c, d]) generatorDevice (ModelStack '[a', b', c', d']) generatorOutputDevice

instance
  HasInitialize (a, b, c, d, e) generatorDevice (a', b', c', d', e') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b, c, d, e]) generatorDevice (ModelStack '[a', b', c', d', e']) generatorOutputDevice

instance
  HasInitialize (a, b, c, d, e, f) generatorDevice (a', b', c', d', e', f') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b, c, d, e, f]) generatorDevice (ModelStack '[a', b', c', d', e', f']) generatorOutputDevice

instance
  HasInitialize (a, b, c, d, e, f, g) generatorDevice (a', b', c', d', e', f', g') generatorOutputDevice =>
  HasInitialize (ModelStack '[a, b, c, d, e, f, g]) generatorDevice (ModelStack '[a', b', c', d', e', f', g']) generatorOutputDevice

data VectorSpec (n :: Nat) (a :: Type) where
  VectorSpec ::
    forall n a.
    SNat n ->
    VS.Vector n (ModelSpec a) ->
    VectorSpec n a

deriving stock instance Show (ModelSpec a) => Show (VectorSpec n a)

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

type HasStateDict :: Type -> Constraint
class HasStateDict model where
  fromStateDict ::
    forall m.
    (MonadIO m, MonadThrow m, MonadState StateDict m) =>
    ModelSpec model ->
    StateDictKey ->
    m model
  default fromStateDict ::
    forall m.
    ( MonadIO m,
      MonadThrow m,
      MonadState StateDict m,
      Generic model,
      Generic (ModelSpec model),
      GHasStateDict (Rep model) (Rep (ModelSpec model))
    ) =>
    ModelSpec model ->
    StateDictKey ->
    m model
  fromStateDict modelSpec k = to <$> gFromStateDict (from modelSpec) k

  toStateDict ::
    forall m.
    (MonadThrow m, MonadState StateDict m) =>
    StateDictKey ->
    model ->
    m ()
  default toStateDict ::
    forall m.
    ( MonadThrow m,
      MonadState StateDict m,
      Generic model,
      GHasStateDict (Rep model) (Rep (ModelSpec model))
    ) =>
    StateDictKey ->
    model ->
    m ()
  toStateDict k model = gToStateDict k (from model)

type GHasStateDict :: (Type -> Type) -> (Type -> Type) -> Constraint
class GHasStateDict gModel gModelSpec | gModelSpec -> gModel, gModel -> gModelSpec where
  gFromStateDict ::
    forall m c.
    (MonadIO m, MonadThrow m, MonadState StateDict m) =>
    gModelSpec c ->
    StateDictKey ->
    m (gModel c)
  gToStateDict ::
    forall m c.
    (MonadThrow m, MonadState StateDict m) =>
    StateDictKey ->
    gModel c ->
    m ()

instance
  (GHasStateDict gModelA gModelSpecA, GHasStateDict gModelB gModelSpecB) =>
  GHasStateDict
    (gModelA :*: gModelB)
    (gModelSpecA :*: gModelSpecB)
  where
  gFromStateDict (gModelASpec :*: gModelBSpec) k =
    (:*:) <$> gFromStateDict gModelASpec k <*> gFromStateDict gModelBSpec k
  gToStateDict k (gModelA :*: gModelB) = do
    () <- gToStateDict k gModelA
    () <- gToStateDict k gModelB
    pure ()

instance
  GHasStateDict gModel gModelSpec =>
  GHasStateDict (M1 i t gModel) (M1 i t gModelSpec)
  where
  gFromStateDict (M1 gModelSpec) k =
    M1 <$> gFromStateDict gModelSpec k
  gToStateDict k (M1 gModel) =
    gToStateDict k gModel

instance
  (HasStateDict model, modelSpec ~ ModelSpec model) =>
  GHasStateDict (K1 i model) (K1 i modelSpec)
  where
  gFromStateDict (K1 modelSpec) k =
    K1 <$> fromStateDict modelSpec k
  gToStateDict k (K1 model) =
    toStateDict k model

instance GHasStateDict U1 U1 where
  gFromStateDict U1 _ = pure U1
  gToStateDict _ U1 = pure ()

instance HasStateDict (SDim dim) where
  fromStateDict dim _ = pure dim
  toStateDict _ _ = pure ()

instance HasStateDict () where
  fromStateDict () _ = pure ()
  toStateDict _ () = pure ()

instance HasStateDict model => HasStateDict (NamedModel model) where
  fromStateDict (NamedModel modelName modelSpec) key =
    NamedModel modelName <$> fromStateDict modelSpec (key <> modelName)
  toStateDict key (NamedModel modelName model) =
    toStateDict (key <> modelName) model

instance
  ( HasStateDict a,
    HasStateDict b
  ) =>
  HasStateDict (a, b)

instance
  ( HasStateDict a,
    HasStateDict b,
    HasStateDict c
  ) =>
  HasStateDict (a, b, c)

instance
  ( HasStateDict a,
    HasStateDict b,
    HasStateDict c,
    HasStateDict d
  ) =>
  HasStateDict (a, b, c, d)

instance
  ( HasStateDict a,
    HasStateDict b,
    HasStateDict c,
    HasStateDict d,
    HasStateDict e
  ) =>
  HasStateDict (a, b, c, d, e)

instance
  ( HasStateDict a,
    HasStateDict b,
    HasStateDict c,
    HasStateDict d,
    HasStateDict e,
    HasStateDict f
  ) =>
  HasStateDict (a, b, c, d, e, f)

instance
  ( HasStateDict a,
    HasStateDict b,
    HasStateDict c,
    HasStateDict d,
    HasStateDict e,
    HasStateDict f,
    HasStateDict g
  ) =>
  HasStateDict (a, b, c, d, e, f, g)

instance HasStateDict (ModelStack '[])

instance
  HasStateDict a =>
  HasStateDict (ModelStack '[a])

instance
  HasStateDict (a, b) =>
  HasStateDict (ModelStack '[a, b])

instance
  HasStateDict (a, b, c) =>
  HasStateDict (ModelStack '[a, b, c])

instance
  HasStateDict (a, b, c, d) =>
  HasStateDict (ModelStack '[a, b, c, d])

instance
  HasStateDict (a, b, c, d, e) =>
  HasStateDict (ModelStack '[a, b, c, d, e])

instance
  HasStateDict (a, b, c, d, e, f) =>
  HasStateDict (ModelStack '[a, b, c, d, e, f])

instance
  HasStateDict (a, b, c, d, e, f, g) =>
  HasStateDict (ModelStack '[a, b, c, d, e, f, g])

type instance ModelSpec (Tensor gradient layout device dataType shape) = TensorSpec gradient layout device dataType shape

instance
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
