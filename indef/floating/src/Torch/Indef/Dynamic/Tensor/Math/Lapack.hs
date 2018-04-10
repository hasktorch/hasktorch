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
  _getri :: Dynamic -> Dynamic -> IO ()
  _getri a b = with2DynamicState a b Sig.c_getri

  _potrf :: Dynamic -> Dynamic -> [Int8] -> IO ()
  _potrf a b c = mkCCharArray c >>= \c' -> with2DynamicState a b $ shuffle3 Sig.c_potrf c'

  _potri :: Dynamic -> Dynamic -> [Int8] -> IO ()
  _potri a b v =
    mkCCharArray v >>= \v' ->
      with2DynamicState a b $ shuffle3 Sig.c_potri v'

  _potrs :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  _potrs a b c v =
    mkCCharArray v >>= \v' ->
      with3DynamicState a b c $ \s' a' b' c' -> Sig.c_potrs s' a' b' c' v'

  _geqrf :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _geqrf a b c = with3DynamicState a b c Sig.c_geqrf

  _qr :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _qr a b c = with3DynamicState a b c Sig.c_qr

  _geev :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  _geev a b c v =
    mkCCharArray v >>= \v' ->
      with3DynamicState a b c $ \s' a' b' c' -> Sig.c_geev s' a' b' c' v'

  _syev :: Dynamic -> Dynamic -> Dynamic -> [Int8] -> [Int8] -> IO ()
  _syev a b c v v0 = do
    v'  <- mkCCharArray v
    v0' <- mkCCharArray v0
    with3DynamicState a b c $ \s' a' b' c' -> Sig.c_syev s' a' b' c' v' v0'

  _gesv :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
  _gesv a b c d =
    with2DynamicState a b $ \s' a' b' ->
      with2DynamicState c d $ \_ c' d' ->
        Sig.c_gesv s' a' b' c' d'

  _gels      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
  _gels a b c d =
    with2DynamicState a b $ \s' a' b' ->
      with2DynamicState c d $ \_ c' d' ->
        Sig.c_gels s' a' b' c' d'

  _gesvd     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  _gesvd a b c d v =
    mkCCharArray v >>= \v' ->
      with2DynamicState a b $ \s' a' b' ->
        with2DynamicState c d $ \_ c' d' ->
          Sig.c_gesvd s' a' b' c' d' v'

  _gesvd2    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> [Int8] -> IO ()
  _gesvd2 a b c d e v = do
    v'  <- mkCCharArray v
    with2DynamicState a b $ \s' a' b' ->
      with3DynamicState c d e $ \_ c' d' e' ->
        Sig.c_gesvd2 s' a' b' c' d' e' v'


