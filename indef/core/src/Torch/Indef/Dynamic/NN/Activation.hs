-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.NN.Activation
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- DYNAMIC-NN MODULE WARNING: this module is mostly unfinished and
-- undocumented. It provides, in essence, direct calls to the torch neural
-- network libraries: THNN and THCUNN. Because the dynamic tensor code requires
-- a lot of runtime checks which requires a lot of thought regarding a good
-- API, the recommended route is to use Static tensors, which have a much more
-- natural API and is inherently safer.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.NN.Activation where

import Torch.Indef.Types
import qualified Torch.Sig.NN as Sig

-- | threshold forward pass (updates the output tensor)
_threshold_updateOutput :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_threshold_updateOutput inp out threshold val inplace = with2DynamicState inp out $ \s' inp' out' ->
  Sig.c_Threshold_updateOutput s' inp' out' (realToFrac threshold) (realToFrac val) (toEnum $ fromEnum inplace)

-- | threshold backward-update (updates the layer and bias tensors)
_threshold_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_threshold_updateGradInput inp gout gin threshold val inplace =  with3DynamicState inp gout gin $ \s' inp' gout' gin' ->
  Sig.c_Threshold_updateGradInput s' inp' gout' gin' (realToFrac threshold) (realToFrac val) (toEnum $ fromEnum inplace)


-- | pReLU forward pass (updates the output tensor)
_pReLU_updateOutput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_pReLU_updateOutput inp gout gin =
  with3DynamicState inp gout gin $ \s' inp' gout' gin' ->
    Sig.c_PReLU_updateOutput s' inp' gout' gin'

-- | pReLU backward-update (updates the layer and bias tensors)
_pReLU_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_pReLU_updateGradInput t0 t1 t2 t3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \ _ t2' t3' ->
      Sig.c_PReLU_updateGradInput s' t0' t1' t2' t3'


-- | pReLU backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_pReLU_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
_pReLU_accGradParameters t0 t1 t2 t3 t4 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    Sig.c_PReLU_accGradParameters s' t0' t1' t2' t3' t4' (realToFrac d0)


-- | rReLU forward pass (updates the output tensor)
_rReLU_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> Bool -> Generator -> IO ()
_rReLU_updateOutput t0 t1 t2 d0 d1 b0 b1 g =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    -- withGen g $ \g' ->
      Sig.c_RReLU_updateOutput s' t0' t1' t2'
        (realToFrac d0) (realToFrac d1)
        (toEnum $ fromEnum b0) (toEnum $ fromEnum b1) undefined -- g'

-- | rReLU backward-update (updates the layer and bias tensors)
_rReLU_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> Bool -> IO ()
_rReLU_updateGradInput t0 t1 t2 t3 d0 d1 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_  t2' t3' ->
      Sig.c_RReLU_updateGradInput s' t0' t1' t2' t3'
        (realToFrac d0) (realToFrac d1)
        (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)


-- | leakyReLU forward pass (updates the output tensor)
_leakyReLU_updateOutput :: Dynamic -> Dynamic -> Double -> Bool -> IO ()
_leakyReLU_updateOutput t0 t1 d0 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
      Sig.c_LeakyReLU_updateOutput s' t0' t1'
        (realToFrac d0) (toEnum $ fromEnum b0)


-- | leakyReLU backward-update (updates the layer and bias tensors)
_leakyReLU_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Double -> Bool -> IO ()
_leakyReLU_updateGradInput t0 t1 t2 d0 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
      Sig.c_LeakyReLU_updateGradInput s' t0' t1' t1'
        (realToFrac d0) (toEnum $ fromEnum b0)

-- | eLU forward pass (updates the output tensor)
_eLU_updateOutput :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_eLU_updateOutput t0 t1 d0 d1 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
      Sig.c_ELU_updateOutput s' t0' t1'
        (realToFrac d0) (realToFrac d1)
        (toEnum $ fromEnum b0) 

-- | eLU backward-update (updates the layer and bias tensors)
_eLU_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
_eLU_updateGradInput t0 t1 t2 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
      Sig.c_ELU_updateGradInput s' t0' t1' t2'
        (realToFrac d0) (realToFrac d1)


