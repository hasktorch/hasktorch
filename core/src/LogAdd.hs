module LogAdd (
  logAdd
  , logSub
  , expMinusApprox
  ) where

import THLogAdd
import System.IO.Unsafe (unsafePerformIO)

logAdd :: Double -> Double -> Double
logAdd log_a log_b =
  realToFrac $ c_THLogAdd log_aC log_aB
  where
    (log_aC, log_aB) = (realToFrac log_a, realToFrac log_b)

logSub :: Double -> Double -> Double
logSub log_a log_b =
  if log_a < log_b then
    (unsafePerformIO $ (fail "log_a must be greater than log_b") >> pure 0) else
    realToFrac $ c_THLogSub log_aC log_aB
  where
    (log_aC, log_aB) = (realToFrac log_a, realToFrac log_b)

expMinusApprox :: Double -> Double
expMinusApprox x =
  realToFrac . c_THExpMinusApprox $ xC
  where
    xC = realToFrac x

test :: IO ()
test = do
  print $ logAdd (-3.0) (-2.0)
  print $ logAdd (-3.0) (2.0)
  -- print $ logSub (-3.0) (-2.0)
  print $ logSub (3.0) (-2.0)
  -- print $ logSub (-3.0) (2.0)
  print $ expMinusApprox (3.0)
  print $ expMinusApprox (-3.0)
  pure ()
