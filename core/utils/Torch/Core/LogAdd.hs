-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.LogAdd
-- Copyright :  (c) Hasktorch devs 2017
-- License   :  BSD3
-- Maintainer:  Sam Stites <sam@stites.io>
-- Stability :  experimental
-- Portability: non-portable
--
-- Various bindings to 'TH/THLogAdd.c' and haskell variants where possible
-------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.LogAdd
  ( logAdd
  , logSub
  , expMinusApprox
  ) where

import Torch.Core.Exceptions
import qualified Torch.FFI.TH.LogAdd as TH

-- | Add two log values, calling out to TH
logAdd :: Double -> Double -> IO Double
logAdd a b = realToFrac <$> TH.c_THLogAdd (realToFrac a) (realToFrac b)

-- | Subtract two log values, calling out to TH
logSub :: Double -> Double -> IO Double
logSub log_a log_b
  | log_a < log_b = throw $ MathException "log_a must be greater than log_b"
  | otherwise     = realToFrac <$> TH.c_THLogSub (realToFrac log_a) (realToFrac log_b)

-- | A fast approximation of @exp(-x)@ for positive @x@. Calls out to TH
expMinusApprox :: Double -> IO Double
expMinusApprox a = realToFrac <$> TH.c_THExpMinusApprox (realToFrac a)

-- | A pure version of 'expMinusApprox', transcribing the code from THLogAdd.c to haskell
expMinusApprox' :: forall f . RealFrac f => f -> Maybe f
expMinusApprox' x
  | x < 0     = Nothing
  | x < 13    = Just $ 1 / (y*y*y*y)
  | otherwise = Just   0
  where
   a0, a1, a2, a3, a4 :: f
   a0 = 1
   a1 = 0.125
   a2 = 0.0078125
   a3 = 0.00032552083
   a4 = 1.0172526e-5
   y  = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
