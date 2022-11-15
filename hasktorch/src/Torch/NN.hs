{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}

module Torch.NN where

import Control.Applicative (Applicative (liftA2))
import Control.Monad.State.Strict
import Data.Foldable (toList)
import Data.Kind
import GHC.Generics
import System.IO.Unsafe (unsafePerformIO)
import Torch.Autograd
import Torch.Device
import Torch.Functional
import Torch.Initializers
import Torch.Internal.Cast (cast3)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import Torch.Scalar
import Torch.Tensor
import Torch.TensorFactories (ones', randIO', randnIO', zeros')

type Parameter = IndependentTensor

type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class HasForward f a b | f a -> b where
  forward :: f -> a -> b
  default forward ::
    ( Generic f,
      Generic a,
      Generic b,
      GHasForward (Rep f) (Rep a) (Rep b)
    ) =>
    f ->
    a ->
    b
  forward f a = to $ gForward (from f) (from a)
  forwardStoch :: f -> a -> IO b
  default forwardStoch ::
    ( Generic f,
      Generic a,
      Generic b,
      GHasForward (Rep f) (Rep a) (Rep b)
    ) =>
    f ->
    a ->
    IO b
  forwardStoch f a = to <$> gForwardStoch (from f) (from a)

class GHasForward (f :: Type -> Type) (a :: Type -> Type) (b :: Type -> Type) | f a -> b where
  gForward :: forall c c' c''. f c -> a c' -> b c''
  gForwardStoch :: forall c c' c''. f c -> a c' -> IO (b c)

instance GHasForward U1 U1 U1 where
  gForward U1 U1 = U1
  gForwardStoch U1 U1 = return U1

instance
  ( GHasForward f a b,
    GHasForward g a' b',
    b'' ~ (b :+: b')
  ) =>
  GHasForward (f :+: g) (a :+: a') b''
  where
  gForward (L1 f) (L1 a) = L1 $ gForward f a
  gForward (R1 g) (R1 a') = R1 $ gForward g a'
  gForwardStoch (L1 f) (L1 a) = L1 <$> gForwardStoch f a
  gForwardStoch (R1 g) (R1 a') = R1 <$> gForwardStoch g a'

instance
  ( GHasForward f a b,
    GHasForward g a' b',
    b'' ~ (b :*: b')
  ) =>
  GHasForward (f :*: g) (a :*: a') b''
  where
  gForward (f :*: g) (a :*: a') = gForward f a :*: gForward g a'
  gForwardStoch (f :*: g) (a :*: a') = liftA2 (:*:) (gForwardStoch f a) (gForwardStoch g a')

instance
  (HasForward f a b) =>
  GHasForward (K1 i f) (K1 i a) (K1 i b)
  where
  gForward (K1 f) (K1 a) = K1 $ forward f a
  gForwardStoch (K1 f) (K1 a) = K1 <$> forwardStoch f a

instance
  (GHasForward f a b) =>
  GHasForward (M1 i t f) (M1 i t' a) (M1 i t' b)
  where
  gForward (M1 f) (M1 a) = M1 $ gForward f a
  gForwardStoch (M1 f) (M1 a) = M1 <$> gForwardStoch f a

class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, GParameterized (Rep f)) => f -> [Parameter]
  flattenParameters f = gFlattenParameters (from f)

  _replaceParameters :: f -> ParamStream f
  default _replaceParameters :: (Generic f, GParameterized (Rep f)) => f -> ParamStream f
  _replaceParameters f = to <$> _gReplaceParameters (from f)

replaceParameters :: Parameterized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (_replaceParameters f) params
   in if null remaining
        then f'
        else error "Some parameters in a call to replaceParameters haven't been consumed!"

instance Parameterized Tensor where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Parameter where
  flattenParameters = pure
  _replaceParameters _ = nextParameter

instance {-# OVERLAPS #-} (Scalar a) => Parameterized a where
  flattenParameters _ = []
  _replaceParameters = return

instance {-# OVERLAPS #-} (Parameterized a, Parameterized b) => Parameterized (a, b) where
  flattenParameters (a, b) = flattenParameters a ++ flattenParameters b
  _replaceParameters (a, b) = do
    a' <- _replaceParameters a
    b' <- _replaceParameters b
    return (a', b')

instance {-# OVERLAPS #-} (Parameterized a, Parameterized b, Parameterized c) => Parameterized (a, b, c) where
  flattenParameters (a, b, c) = flattenParameters a ++ flattenParameters b ++ flattenParameters c
  _replaceParameters (a, b, c) = do
    a' <- _replaceParameters a
    b' <- _replaceParameters b
    c' <- _replaceParameters c
    return (a', b', c')

instance {-# OVERLAPS #-} (Foldable t, Traversable t, Parameterized a) => Parameterized (t a) where
  flattenParameters = (=<<) flattenParameters . toList
  _replaceParameters = mapM _replaceParameters

instance Parameterized (a -> a) where
  flattenParameters _ = []
  _replaceParameters = return

class GParameterized f where
  gFlattenParameters :: forall a. f a -> [Parameter]
  _gReplaceParameters :: forall a. f a -> ParamStream (f a)

instance GParameterized U1 where
  gFlattenParameters U1 = []
  _gReplaceParameters U1 = return U1

instance (GParameterized f, GParameterized g) => GParameterized (f :+: g) where
  gFlattenParameters (L1 x) = gFlattenParameters x
  gFlattenParameters (R1 x) = gFlattenParameters x
  _gReplaceParameters (L1 x) = do
    x' <- _gReplaceParameters x
    return $ L1 x'
  _gReplaceParameters (R1 x) = do
    x' <- _gReplaceParameters x
    return $ R1 x'

instance (GParameterized f, GParameterized g) => GParameterized (f :*: g) where
  gFlattenParameters (x :*: y) = gFlattenParameters x ++ gFlattenParameters y
  _gReplaceParameters (x :*: y) = do
    x' <- _gReplaceParameters x
    y' <- _gReplaceParameters y
    return $ x' :*: y'

instance (Parameterized c) => GParameterized (K1 i c) where
  gFlattenParameters (K1 x) = flattenParameters x
  _gReplaceParameters (K1 x) = do
    x' <- _replaceParameters x
    return $ K1 x'

instance (GParameterized f) => GParameterized (M1 i t f) where
  gFlattenParameters (M1 x) = gFlattenParameters x
  _gReplaceParameters (M1 x) = do
    x' <- _gReplaceParameters x
    return $ M1 x'

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

--
-- Linear FC Layer
--

data LinearSpec = LinearSpec
  { in_features :: Int,
    out_features :: Int
  }
  deriving (Show, Eq)

data Linear = Linear
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

linear :: Linear -> Tensor -> Tensor
linear layer input = linear' input w b
  where
    linear' input weight bias = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

linearForward :: Linear -> Tensor -> Tensor
linearForward = linear -- temporary alias until dependencies are updated

instance HasForward Linear Tensor Tensor where
  forward = linearForward
  forwardStoch m x = pure $ linearForward m x

instance Randomizable LinearSpec Linear where
  sample LinearSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [out_features, in_features]
    init <- randIO' [out_features]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ out_features,
                          in_features
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ Linear w b

--
-- Conv1d
--
data Conv1dSpec = Conv1dSpec
  { inputChannelSize1d :: Int,
    outputChannelSize1d :: Int,
    kernelSize :: Int
  }
  deriving (Show, Eq)

data Conv1d = Conv1d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

conv1dForward ::
  -- | layer
  Conv1d ->
  -- | stride
  Int ->
  -- | padding
  Int ->
  -- | input
  Tensor ->
  -- | output
  Tensor
conv1dForward layer = Torch.Functional.conv1d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable Conv1dSpec Conv1d where
  sample Conv1dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ outputChannelSize1d,
            inputChannelSize1d,
            kernelSize
          ]
    init <- randIO' [outputChannelSize1d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ outputChannelSize1d,
                          inputChannelSize1d,
                          kernelSize
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ Conv1d w b

--
-- Conv2d
--

data Conv2dSpec = Conv2dSpec
  { inputChannelSize2d :: Int,
    outputChannelSize2d :: Int,
    kernelHeight2d :: Int,
    kernelWidth2d :: Int
  }
  deriving (Show, Eq)

data Conv2d = Conv2d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

conv2dForward ::
  -- | layer
  Conv2d ->
  -- | stride
  (Int, Int) ->
  -- | padding
  (Int, Int) ->
  -- | input
  Tensor ->
  -- | output
  Tensor
conv2dForward layer = Torch.Functional.conv2d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable Conv2dSpec Conv2d where
  sample Conv2dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ outputChannelSize2d,
            inputChannelSize2d,
            kernelHeight2d,
            kernelWidth2d
          ]
    init <- randIO' [outputChannelSize2d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ outputChannelSize2d,
                          inputChannelSize2d,
                          kernelHeight2d,
                          kernelWidth2d
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ Conv2d w b

--
-- Conv3d
--

data Conv3dSpec = Conv3dSpec
  { inputChannelSize3d :: Int,
    outputChannelSize3d :: Int,
    kernelHeight3d :: Int,
    kernelWidth3d :: Int,
    kernelDepth3d :: Int
  }
  deriving (Show, Eq)

data Conv3d = Conv3d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

conv3dForward ::
  -- | layer
  Conv3d ->
  -- | stride
  (Int, Int, Int) ->
  -- | padding
  (Int, Int, Int) ->
  -- | input
  Tensor ->
  -- | output
  Tensor
conv3dForward layer = Torch.Functional.conv3d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable Conv3dSpec Conv3d where
  sample Conv3dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ outputChannelSize3d,
            inputChannelSize3d,
            kernelHeight3d,
            kernelWidth3d,
            kernelDepth3d
          ]
    init <- randIO' [outputChannelSize3d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ outputChannelSize3d,
                          inputChannelSize3d,
                          kernelHeight3d,
                          kernelWidth3d,
                          kernelDepth3d
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ Conv3d w b

--
-- ConvTranspose1d
--

data ConvTranspose1dSpec = ConvTranspose1dSpec
  { trInputChannelSize1d :: Int,
    trOutputChannelSize1d :: Int,
    trKernelSize :: Int
  }
  deriving (Show, Eq)

data ConvTranspose1d = ConvTranspose1d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

convTranspose1dForward ::
  -- | layer
  ConvTranspose1d ->
  -- | stride
  Int ->
  -- | padding
  Int ->
  -- | input
  Tensor ->
  -- | output
  Tensor
convTranspose1dForward layer = convTranspose1d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable ConvTranspose1dSpec ConvTranspose1d where
  sample ConvTranspose1dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ trInputChannelSize1d,
            trOutputChannelSize1d,
            trKernelSize
          ]
    init <- randIO' [trOutputChannelSize1d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ trInputChannelSize1d,
                          trOutputChannelSize1d,
                          trKernelSize
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ ConvTranspose1d w b

--
-- ConvTranspose2d
--

data ConvTranspose2dSpec = ConvTranspose2dSpec
  { trInputChannelSize2d :: Int,
    trOutputChannelSize2d :: Int,
    trKernelHeight2d :: Int,
    trKernelWidth2d :: Int
  }
  deriving (Show, Eq)

data ConvTranspose2d = ConvTranspose2d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

convTranspose2dForward ::
  -- | layer
  ConvTranspose2d ->
  -- | stride
  (Int, Int) ->
  -- | padding
  (Int, Int) ->
  -- | input
  Tensor ->
  -- | output
  Tensor
convTranspose2dForward layer = convTranspose2d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable ConvTranspose2dSpec ConvTranspose2d where
  sample ConvTranspose2dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ trInputChannelSize2d,
            trOutputChannelSize2d,
            trKernelHeight2d,
            trKernelWidth2d
          ]
    init <- randIO' [trOutputChannelSize2d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ trInputChannelSize2d,
                          trOutputChannelSize2d,
                          trKernelHeight2d,
                          trKernelWidth2d
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ ConvTranspose2d w b

--
-- ConvTranspose2d
--

data ConvTranspose3dSpec = ConvTranspose3dSpec
  { trInputChannelSize3d :: Int,
    trOutputChannelSize3d :: Int,
    trKernelHeight3d :: Int,
    trKernelWidth3d :: Int,
    trKernelDepth3d :: Int
  }
  deriving (Show, Eq)

data ConvTranspose3d = ConvTranspose3d
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

convTranspose3dForward ::
  -- | layer
  ConvTranspose3d ->
  -- | stride
  (Int, Int, Int) ->
  -- | padding
  (Int, Int, Int) ->
  -- | input
  Tensor ->
  -- | output
  Tensor
convTranspose3dForward layer = convTranspose3d' w b
  where
    w = toDependent (layer.weight)
    b = toDependent (layer.bias)

instance Randomizable ConvTranspose3dSpec ConvTranspose3d where
  sample ConvTranspose3dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ trInputChannelSize3d,
            trOutputChannelSize3d,
            trKernelHeight3d,
            trKernelWidth3d,
            trKernelDepth3d
          ]
    init <- randIO' [trOutputChannelSize3d]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ trInputChannelSize3d,
                          trOutputChannelSize3d,
                          trKernelHeight3d,
                          trKernelWidth3d,
                          trKernelDepth3d
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )
    return $ ConvTranspose3d w b

data BatchNormSpec = BatchNormSpec
  { numFeatures :: Int
  }
  deriving (Show, Eq)

data BatchNorm = BatchNorm
  { weight :: Parameter,
    bias :: Parameter,
    runningMean :: MutableTensor,
    runningVar :: MutableTensor
  }
  deriving (Show, Generic)

batchNormForwardIO :: BatchNorm -> Bool -> Double -> Double -> Tensor -> IO Tensor
batchNormForwardIO params train momentum eps input =
  Torch.Functional.batchNormIO
    (toDependent params.weight)
    (toDependent params.bias)
    params.runningMean
    params.runningVar
    train
    momentum
    eps
    input

instance Randomizable BatchNormSpec BatchNorm where
  sample BatchNormSpec {..} = do
    w <- makeIndependent (ones' [numFeatures])
    b <- makeIndependent (zeros' [numFeatures])
    mean <- MutableTensor <$> toDependent <$> makeIndependentWithRequiresGrad (zeros' [numFeatures]) False
    var <- MutableTensor <$> toDependent <$> makeIndependentWithRequiresGrad (ones' [numFeatures]) False
    return $ BatchNorm w b mean var

data InstanceNormSpec = InstanceNormSpec
  { numFeatures :: Int
  }
  deriving (Show, Eq)

data InstanceNorm = InstanceNorm
  { weight :: Parameter,
    bias :: Parameter,
    runningMean :: MutableTensor,
    runningVar :: MutableTensor
  }
  deriving (Show, Generic)

instanceNormForwardIO :: InstanceNorm -> Bool -> Double -> Double -> Tensor -> IO Tensor
instanceNormForwardIO params train momentum eps input =
  Torch.Functional.instanceNormIO
    (toDependent params.weight)
    (toDependent params.bias)
    params.runningMean
    params.runningVar
    train
    momentum
    eps
    input

instance Randomizable InstanceNormSpec InstanceNorm where
  sample InstanceNormSpec {..} = do
    w <- makeIndependent (ones' [numFeatures])
    b <- makeIndependent (zeros' [numFeatures])
    mean <- MutableTensor <$> toDependent <$> makeIndependentWithRequiresGrad (zeros' [numFeatures]) False
    var <- MutableTensor <$> toDependent <$> makeIndependentWithRequiresGrad (ones' [numFeatures]) False
    return $ InstanceNorm w b mean var

data UpSampleSpec = UpSampleSpec
  { upsampleInputFilters :: Int,
    upsampleStride :: Int
  }
  deriving (Show, Eq)

instance Parameterized UpSampleSpec where
  flattenParameters _ = []
  _replaceParameters = return

data UpSample = UpSample
  { upsampleSpec :: UpSampleSpec
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable UpSampleSpec UpSample where
  sample s = do
    UpSample
      <$> pure s

instance HasForward UpSample Tensor Tensor where
  forward (UpSample (UpSampleSpec {..})) input =
    upsampleNearest2d (outputWidth * upsampleStride, outputHeight * upsampleStride) (fromIntegral upsampleStride) (fromIntegral upsampleStride) input
    where
      outputWidth : outputHeight : _ = reverse $ shape input
  forwardStoch m x = pure $ forward m x
