{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.NN where

import Control.Applicative (Applicative (liftA2))
import Control.Monad.State.Strict
import Data.Kind
import GHC.Generics
import System.IO.Unsafe (unsafePerformIO)
import Torch.Autograd
import Torch.Functional
import Torch.Initializers
import Torch.Internal.Cast (cast3)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import Torch.Tensor
import Torch.TensorFactories (ones', randIO', randnIO')

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
  gForwardStoch (f :*: g) (a :*: a') = liftA2 ((:*:)) (gForwardStoch f a) (gForwardStoch g a')

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

instance Parameterized Int where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Float where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Double where
  flattenParameters _ = []
  _replaceParameters = return

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

instance Parameterized [Linear]

linear :: Linear -> Tensor -> Tensor
linear layer input = linear' input w b
  where
    linear' input weight bias = unsafePerformIO $ (cast3 ATen.linear_ttt) input weight bias
    w = toDependent (weight layer)
    b = toDependent (bias layer)

linearForward :: Linear -> Tensor -> Tensor
linearForward = linear -- temporary alias until dependencies are updated

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
-- Conv2d
--

data Conv2dSpec = Conv2dSpec
  { inputChannelSize :: Int,
    outputChannelSize :: Int,
    kernelHeight :: Int,
    kernelWidth :: Int
  }
  deriving (Show, Eq)

data Conv2d = Conv2d
  { conv2dWeight :: Parameter,
    conv2dBias :: Parameter
  }
  deriving (Show, Generic)

conv2dForward :: Conv2d -> (Int, Int) -> (Int, Int) -> Tensor -> Tensor
conv2dForward layer stride padding input =
  Torch.Functional.conv2d' w b stride padding input
  where
    w = toDependent (conv2dWeight layer)
    b = toDependent (conv2dBias layer)

instance Randomizable Conv2dSpec Conv2d where
  sample Conv2dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ outputChannelSize,
            inputChannelSize,
            kernelHeight,
            kernelWidth
          ]
    init <- randIO' [outputChannelSize]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ outputChannelSize,
                          inputChannelSize,
                          kernelHeight,
                          kernelWidth
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
