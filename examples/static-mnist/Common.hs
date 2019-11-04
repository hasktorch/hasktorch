{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}

module Common where

import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe
import           System.Random

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Native
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D
import qualified Image                         as I

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123

toBackend
  :: forall t . (ATen.Castable t (ForeignPtr ATen.Tensor)) => String -> t -> t
toBackend backend t = unsafePerformIO $ case backend of
  "CUDA" -> ATen.cast1 ATen.tensor_cuda t
  _      -> ATen.cast1 ATen.tensor_cpu t

crossEntropyLoss
  :: forall batchSize seqLen dtype device
   . ( KnownNat batchSize
     , KnownNat seqLen
     , KnownDType dtype
     , KnownDevice device
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype '[batchSize, seqLen]
  -> Tensor device 'D.Int64 '[batchSize]
  -> Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @seqLen @'[]
    ones
    (-100)
    (logSoftmax @1 prediction)
    target

errorRate
  :: forall batchSize outputFeatures device
   . ( KnownNat batchSize
     , KnownNat outputFeatures
     , SumDTypeIsValid device 'D.Bool
     , ComparisonDTypeIsValid device 'D.Int64
     )
  => Tensor device 'D.Float '[batchSize, outputFeatures]
  -> Tensor device 'D.Int64 '[batchSize]
  -> Tensor device 'D.Float '[]
errorRate prediction target =
  let errorCount =
          toDType @D.Float . sumAll . ne (argmax @1 @DropDim prediction) $ target
  in  cmul ((1.0 /) . fromIntegral $ natValI @batchSize :: Double) errorCount

withTestSize
  :: Int -> (forall testSize . KnownNat testSize => Proxy testSize -> a) -> a
withTestSize nat fn = case someNatVal (fromIntegral nat) of
  Just (SomeNat (Proxy :: Proxy testSize)) -> fn (Proxy @testSize)
  _ -> error "Cannot get the size of the test dataset"
