module Torch.Indef.Dynamic.Tensor.Math.Lapack () where

import GHC.Int
import Data.Coerce
import Foreign
import Foreign.C.Types
import Foreign.Marshal.Array
import qualified Torch.Class.Tensor.Math.Lapack as Class
import qualified Torch.Sig.Tensor.Math.Lapack as Sig

import Torch.Indef.Types

mkCCharArray :: [Int8] -> IO (Ptr CChar)
mkCCharArray c = newArray (coerce c :: [CChar])

instance Class.TensorMathLapack Dynamic where
  getri_ :: Dynamic -> Dynamic -> IO ()
  getri_ a b = with2DynamicState a b Sig.c_getri

  potrf_ :: Dynamic -> Dynamic -> [Int8] -> IO ()
  potrf_ a b c = mkCCharArray c >>= \c' -> with2DynamicState a b $ shuffle3 Sig.c_potrf c'

  potri_ :: Dynamic -> Dynamic -> [Int8] -> IO ()
  potri_ a b v =
    mkCCharArray v >>= \v' ->
      with2DynamicState a b $ shuffle3 Sig.c_potri v'

  potrs_ :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  potrs_ a b c v =
    mkCCharArray v >>= \v' ->
      with3DynamicState a b c $ \s' a' b' c' -> Sig.c_potrs s' a' b' c' v'

  geqrf_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  geqrf_ a b c = with3DynamicState a b c Sig.c_geqrf

  qr_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  qr_ a b c = with3DynamicState a b c Sig.c_qr

  geev_ :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  geev_ a b c v =
    mkCCharArray v >>= \v' ->
      with3DynamicState a b c $ \s' a' b' c' -> Sig.c_geev s' a' b' c' v'

  syev_ :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> [Int8] -> IO ()
  syev_ a b c v v0 = do
    v'  <- mkCCharArray v
    v0' <- mkCCharArray v0
    with3DynamicState a b c $ \s' a' b' c' -> Sig.c_syev s' a' b' c' v' v0'

  gesv_ :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
  gesv_ a b c d =
    with2DynamicState a b $ \s' a' b' ->
      with2DynamicState c d $ \_ c' d' ->
        Sig.c_gesv s' a' b' c' d'

  gels_      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
  gels_ a b c d =
    with2DynamicState a b $ \s' a' b' ->
      with2DynamicState c d $ \_ c' d' ->
        Sig.c_gels s' a' b' c' d'

  gesvd_     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  gesvd_ a b c d v =
    mkCCharArray v >>= \v' ->
      with2DynamicState a b $ \s' a' b' ->
        with2DynamicState c d $ \_ c' d' ->
          Sig.c_gesvd s' a' b' c' d' v'

  gesvd2_    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  gesvd2_ a b c d e v = do
    v'  <- mkCCharArray v
    with2DynamicState a b $ \s' a' b' ->
      with3DynamicState c d e $ \_ c' d' e' ->
        Sig.c_gesvd2 s' a' b' c' d' e' v'


