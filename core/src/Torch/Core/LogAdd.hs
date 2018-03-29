{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.LogAdd
  ( logAdd
  , logSub
  , expMinusApprox
  ) where

import Torch.Core.Exceptions
import qualified THLogAdd as TH

-- | Add two log values, calling out to TH
logAdd :: Double -> Double -> IO Double
logAdd a b = pure . realToFrac $ TH.c_THLogAdd (realToFrac a) (realToFrac b)

-- | Subtract two log values, calling out to TH
logSub :: Double -> Double -> IO Double
logSub log_a log_b
  | log_a < log_b = throw $ MathException "log_a must be greater than log_b"
  | otherwise     = pure . realToFrac $ TH.c_THLogSub (realToFrac log_a) (realToFrac log_b)

expMinusApprox :: Double -> IO Double
expMinusApprox a = pure . realToFrac $ TH.c_THExpMinusApprox (realToFrac a)


