-- | Garbage-collection policy applied before each optimizer step.
--
-- 'runStep'' forces a major GC before every update so that tensors which are
-- only referenced from the Haskell heap release their libtorch memory in time.
-- Forcing a major GC on every step has a cost: each call traverses the whole
-- live Haskell heap, so with a large live heap the total cost grows with
-- @steps * liveHeap@ (see <https://gitlab.haskell.org/ghc/ghc/-/issues/18239>).
-- Even with a small heap the per-step overhead is measurable (about 10% for an
-- MLP training loop). 'GcPolicy' makes that trade-off explicit.
--
-- The default policy used by 'Torch.Optim.runStep'' is 'GcEvery', which keeps
-- the historical behaviour. It can be overridden without code changes by
-- setting the environment variable @HASKTORCH_GC_INTERVAL_MS@:
--
-- * unset, or not a non-negative integer: 'GcEvery'
-- * @0@: 'GcNever'
-- * @N > 0@: at most one forced GC every @N@ milliseconds
module Torch.Optim.GC
  ( GcPolicy (..),
    GcInterval,
    gcIntervalMilliseconds,
    newGcInterval,
    gcPolicyFromEnv,
    defaultGcPolicy,
    runGcPolicy,
    gcIntervalEnvVar,
  )
where

import Data.IORef (IORef, atomicModifyIORef', newIORef)
import Data.Word (Word64)
import GHC.Clock (getMonotonicTimeNSec)
import System.Environment (lookupEnv)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC)
import Text.Read (readMaybe)
import Torch.Internal.GC (mallocTrim)

-- | Rate-limiting state for 'GcRateLimited': the minimum distance between two
-- forced collections and the monotonic time of the last one.
data GcInterval = GcInterval
  { gcIntervalNSec :: !Word64,
    gcLastRunNSec :: !(IORef (Maybe Word64))
  }

-- | When to force a major GC (followed by @malloc_trim@) before an optimizer step.
data GcPolicy
  = -- | Force a GC before every step.
    GcEvery
  | -- | Force a GC at most once per interval; steps in between skip it.
    GcRateLimited !GcInterval
  | -- | Never force a GC; rely on the RTS and tensor finalizers.
    GcNever

-- | Minimum interval of a rate-limited policy, in milliseconds.
gcIntervalMilliseconds :: GcInterval -> Word64
gcIntervalMilliseconds = (`div` nsecPerMsec) . gcIntervalNSec

-- | Create a rate-limited policy that forces at most one GC per given number
-- of milliseconds. The first step always collects.
newGcInterval :: Word64 -> IO GcPolicy
newGcInterval ms = GcRateLimited . GcInterval (ms * nsecPerMsec) <$> newIORef Nothing

-- | Name of the environment variable read by 'gcPolicyFromEnv'.
gcIntervalEnvVar :: String
gcIntervalEnvVar = "HASKTORCH_GC_INTERVAL_MS"

-- | Build a policy from @HASKTORCH_GC_INTERVAL_MS@ (see the module header).
gcPolicyFromEnv :: IO GcPolicy
gcPolicyFromEnv = do
  value <- lookupEnv gcIntervalEnvVar
  case value >>= readMaybe :: Maybe Integer of
    Just 0 -> pure GcNever
    Just ms | ms > 0 -> newGcInterval (fromInteger ms)
    _ -> pure GcEvery

-- | Process-wide policy used by 'Torch.Optim.runStep'' and the other built-in
-- step functions. The environment is read once, on first use.
--
-- This is a top-level mutable cell created with 'unsafePerformIO'. The
-- @NOINLINE@ pragma guarantees a single shared instance; without it GHC could
-- duplicate the CAF and each copy would keep its own rate-limiting timer.
defaultGcPolicy :: GcPolicy
defaultGcPolicy = unsafePerformIO gcPolicyFromEnv
{-# NOINLINE defaultGcPolicy #-}

-- | Apply a policy: force a major GC and trim the C heap when the policy says so.
runGcPolicy :: GcPolicy -> IO ()
runGcPolicy GcEvery = collect
runGcPolicy GcNever = pure ()
runGcPolicy (GcRateLimited (GcInterval interval lastRef)) = do
  now <- getMonotonicTimeNSec
  due <- atomicModifyIORef' lastRef $ \lastRun -> case lastRun of
    Just t | now - t < interval -> (lastRun, False)
    _ -> (Just now, True)
  if due then collect else pure ()

collect :: IO ()
collect = performGC >> mallocTrim 0

nsecPerMsec :: Word64
nsecPerMsec = 1000000
