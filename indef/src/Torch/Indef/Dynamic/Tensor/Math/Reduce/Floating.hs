-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating
  ( mean, mean_
  , std, std_
  , var, var_
  , norm, norm_
  , renorm, renorm_
  , dist
  , meanall
  , varall
  , stdall
  , normall
  ) where

import Control.Monad.Managed
import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor (empty)

import qualified Torch.Sig.Tensor.Math.Reduce.Floating as Sig

-- | Take the mean in the specified dimension.
mean :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> Dynamic
mean t d = unsafeDupablePerformIO $ do
  let r = empty
  _meanKeepDims r t d
  pure r
{-# NOINLINE mean #-}

-- | Inplace 'mean'
mean_ :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> IO ()
mean_ t = _meanKeepDims t t

-- | '_mean' with defaulted 'KeepDim' as 'True' indicating that the result
-- tensor will be 'squeeze1d' in the specified dimension.
_meanKeepDims :: Dynamic -> Dynamic
  -> Word  -- ^ dimension to operate over
  -> IO ()
_meanKeepDims r t d = _mean r t d keep

-- | C-style function of 'mean' and 'mean_'. Should not be exported.
_mean :: Dynamic -> Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> IO ()
_mean r t d b = with2DynamicState r t $ \s' r' t' ->
  Sig.c_mean s' r' t'  (fromIntegral d) (fromIntegral $ fromEnum b)

-- | Performs the @std@ operation over the specified dimension. The 'Bool'
-- parameter specifies whether the standard deviation should be used with
-- @n-1@ or @n@. 'False' normalizes by @n-1@, while 'True' normalizes @n@.
std
  :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim
  -> Bool
  -> Dynamic
std t a b c = unsafeDupablePerformIO $ do
  let r = empty
  _std r t a b c
  pure r
{-# NOINLINE std #-}

-- | Infix version of 'std'.
std_ :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> Bool -> IO ()
std_ t = _std t t

-- | C-style function of 'std' and 'std_'. Should not be exported.
_std :: Dynamic -> Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> Bool -> IO ()
_std r t a b c = with2DynamicState r t $ \s' r' t' ->
  Sig.c_std s' r' t' (fromIntegral a) (fromIntegral $ fromEnum b) (toEnum $ fromEnum c)

-- | Get the variance over a tensor in the specified dimension. The 'Bool'
-- parameter specifies whether the standard deviation should be used with
-- @n-1@ or @n@. 'False' normalizes by @n-1@, while 'True' normalizes @n@.
var :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> Bool -> Dynamic
var t a b c = unsafeDupablePerformIO $ do
  let r = empty
  _var r t a b c
  pure r
{-# NOINLINE var #-}

-- | Infix version of 'var'.
var_ :: Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> Bool -> IO ()
var_ t = _var t t

-- | C-style function of 'var' and 'var_'. Should not be exported.
_var :: Dynamic -> Dynamic
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> Bool -> IO ()
_var r t a b c = with2DynamicState r t $ \s' r' t' -> Sig.c_var s' r' t' (fromIntegral a) (fromIntegral $ fromEnum b) (fromIntegral $ fromEnum c)

-- | Return the @p@-norms of the tensor, computed over dimension @dim@.
norm :: Dynamic -> HsReal
  -> Word  -- ^ dimension to operate over
  -> Dynamic
norm t p d = unsafeDupablePerformIO $ do
  let r = empty
  _normKeepDims r t p d
  pure r
{-# NOINLINE norm #-}

-- | Inplace version of 'norm'
norm_ :: Dynamic -> HsReal
  -> Word  -- ^ dimension to operate over
  -> IO ()
norm_ t = _normKeepDims t t

-- | '_norm' with defaulted 'KeepDim' as 'True' indicating that the result
-- tensor will be 'squeeze1d'd in the specified dimension.
_normKeepDims :: Dynamic -> Dynamic -> HsReal
  -> Word  -- ^ dimension to operate over
  -> IO ()
_normKeepDims r t p d = _norm r t p d keep

-- | C-style function of 'norm' and 'norm_'. Should not be exported.
_norm :: Dynamic -> Dynamic -> HsReal
  -> Word  -- ^ dimension to operate over
  -> KeepDim -> IO ()
_norm r t p d k = with2DynamicState r t $ \s' r' t' -> Sig.c_norm s' r' t' (hs2cReal p) (fromIntegral d) (fromIntegral $ fromEnum k)

-- | Renormalizes the sub-Tensors along dimension @dim@ such that they do not
-- exceed norm @maxnorm@.
--
-- Equivalent to the following lua code: @y = torch.renorm(x, p, dim, maxnorm)@.
-- Returns a version of @x@ with @p@-norms lower than maxnorm over non-@dim@
-- dimensions. The @dim@ argument is not to be confused with the argument of the
-- same name in function 'norm'. In this case, the @p@-norm is measured for each
-- @i@-th sub-tensor (lua: @x:select(dim, i)@).
renorm
  :: Dynamic    -- ^ @x@
  -> HsReal     -- ^ @p@
  -> Int        -- ^ @dim@
  -> HsReal     -- ^ @maxnorm@
  -> Dynamic    -- ^ @res@
renorm x p dim mn = unsafeDupablePerformIO $ do
  let res = empty
  _renorm res x p dim mn
  pure res
{-# NOINLINE renorm #-}

-- | inplace version of 'renorm'
renorm_ :: Dynamic -> HsReal -> Int -> HsReal -> IO ()
renorm_ t = _renorm t t

-- | C-style function of 'renorm' and 'renorm_'. Should not be exported.
_renorm :: Dynamic -> Dynamic -> HsReal -> Int -> HsReal -> IO ()
_renorm r t v a v0 = with2DynamicState r t $ \s' r' t' -> Sig.c_renorm s' r' t' (hs2cReal v) (fromIntegral a) (hs2cReal v0)

-- | Returns the @p@-norm of @x - y@.
dist
  :: Dynamic       -- ^ tensor @x@
  -> Dynamic       -- ^ tensor @y@
  -> HsReal        -- ^ @p@
  -> HsAccReal
dist r t v = unsafeDupablePerformIO $ with2DynamicState r t $ \s' r' t' -> fmap c2hsAccReal $ Sig.c_dist s' r' t' (hs2cReal v)
{-# NOINLINE dist #-}

-- | Returns the mean of all elements.
meanall :: Dynamic -> HsAccReal
meanall t = unsafeDupablePerformIO $ flip with (pure . c2hsAccReal) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_meanall s' t'
{-# NOINLINE meanall #-}

-- | Returns the variance of all elements.
varall :: Dynamic -> Int -> HsAccReal
varall t v = unsafeDupablePerformIO $ flip with (pure . c2hsAccReal) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_varall s' t' (fromIntegral v)
{-# NOINLINE varall #-}

-- | Returns the standard deviation of all elements.
stdall :: Dynamic -> Int -> HsAccReal
stdall t v = unsafeDupablePerformIO $ flip with (pure . c2hsAccReal) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_stdall s' t' (fromIntegral v)
{-# NOINLINE stdall #-}

-- | Returns the @p@-norm of all elements.
normall
  :: Dynamic  -- ^ tensor of values to norm over
  -> HsReal   -- ^ @p@
  -> HsAccReal
normall t v = unsafeDupablePerformIO $ flip with (pure . c2hsAccReal) $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_normall s' t' (hs2cReal v)
{-# NOINLINE normall #-}
