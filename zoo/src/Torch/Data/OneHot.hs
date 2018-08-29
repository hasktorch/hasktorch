{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Data.OneHot where

import qualified Data.Vector as V

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double
import qualified Torch.Long as Long
#endif

onehotL
  :: forall c sz
  . (Ord c, Bounded c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => c
  -> LongTensor '[10] -- '[FromEnum (MaxBound c)]
onehotL c
  = Long.unsafeVector
  $ onehot c

onehotT
  :: forall c sz
  . (Ord c, Bounded c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => c
  -> Tensor '[10] -- '[FromEnum (MaxBound c)]
onehotT c
  = unsafeVector
  $ fmap fromIntegral
  $ onehot c

onehot
  :: forall i c
  . (Integral i, Ord c, Bounded c, Enum c)
  => c
  -> [i]
onehot c
  = V.toList
  $ V.generate
    (fromEnum (maxBound :: c) + 1)
    (fromIntegral . fromEnum . (== fromEnum c))

onehotf
  :: forall i c
  . (Fractional i, Ord c, Bounded c, Enum c)
  => c
  -> [i]
onehotf c
  = V.toList
  $ V.generate
    (fromEnum (maxBound :: c) + 1)
    (realToFrac . fromIntegral . fromEnum . (== fromEnum c))


