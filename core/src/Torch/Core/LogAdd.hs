{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.LogAdd
  ( logAdd
  , logSub
  , expMinusApprox
  ) where

import Foreign.C.Types (CDouble)
import THLogAdd
import System.IO.Unsafe (unsafePerformIO)
import Torch.Core.Exceptions
import Torch.Core.Internal (genOp1, genOp2)

-- | Add two log values, calling out to TH
logAdd :: (Real a, Fractional b) => a -> a -> b
logAdd log_a log_b = genOp2 c_THLogAdd log_a log_b


-- | Subtract two log values, calling out to TH
logSub :: (MonadThrow m, Real a, Fractional b) => a -> a -> m b
logSub log_a log_b =
  if log_a < log_b
  then throw (MathException "log_a must be greater than log_b")
  else pure (genOp2 c_THLogSub log_a log_a)


-- | Subtract two log values, calling out to TH
unsafeLogSub :: forall a b . (Real a, Fractional b) => a -> a -> b
unsafeLogSub a b = unsafePerformIO (logSub a b :: IO b)


expMinusApprox :: (Real a, Fractional b) => a -> b
expMinusApprox = genOp1 c_THExpMinusApprox

