module Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating where

import GHC.Int
import qualified Torch.Class.Tensor.Math.Reduce as Class
import qualified Torch.Sig.Tensor.Math.Reduce.Floating as Sig

import Torch.Indef.Types

instance Class.TensorMathReduceFloating Dynamic where
  mean_ :: Dynamic -> Dynamic -> Int -> Int -> IO ()
  mean_ r t a b = with2DynamicState r t $ \s' r' t' -> Sig.c_mean s' r' t' (fromIntegral a) (fromIntegral b)

  std_ :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
  std_ r t a b c = with2DynamicState r t $ \s' r' t' -> Sig.c_std s' r' t' (fromIntegral a) (fromIntegral b) (fromIntegral c)

  var_ :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
  var_ r t a b c = with2DynamicState r t $ \s' r' t' -> Sig.c_var s' r' t' (fromIntegral a) (fromIntegral b) (fromIntegral c)

  norm_ :: Dynamic -> Dynamic -> HsReal -> Int -> Int -> IO ()
  norm_ r t v a b = with2DynamicState r t $ \s' r' t' -> Sig.c_norm s' r' t' (hs2cReal v) (fromIntegral a) (fromIntegral b)

  renorm_ :: Dynamic -> Dynamic -> HsReal -> Int -> HsReal -> IO ()
  renorm_ r t v a v0 = with2DynamicState r t $ \s' r' t' -> Sig.c_renorm s' r' t' (hs2cReal v) (fromIntegral a) (hs2cReal v0)

  dist :: Dynamic -> Dynamic -> HsReal -> IO (HsAccReal)
  dist r t v = with2DynamicState r t $ \s' r' t' -> fmap c2hsAccReal $ Sig.c_dist s' r' t' (hs2cReal v)

  meanall :: Dynamic -> IO (HsAccReal)
  meanall t = withDynamicState t (fmap c2hsAccReal .: Sig.c_meanall)

  varall :: Dynamic -> Int -> IO (HsAccReal)
  varall t v = withDynamicState t $ \s' t' -> c2hsAccReal <$> Sig.c_varall s' t' (fromIntegral v)

  stdall :: Dynamic -> Int -> IO (HsAccReal)
  stdall t v = withDynamicState t $ \s' t' -> c2hsAccReal <$> Sig.c_stdall s' t' (fromIntegral v)

  normall :: Dynamic -> HsReal -> IO (HsAccReal)
  normall t v = withDynamicState t $ \s' t' -> c2hsAccReal <$> Sig.c_normall s' t' (hs2cReal v)



