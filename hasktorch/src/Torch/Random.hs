{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Random
  ( mkGenerator,
    Generator,
    randn,
    randn',
    rand,
    rand',
    randint,
    randint',
    normal,
    normal',
    PureGenerator,
    mkPureGenerator,
    withPureGenerator,
    multinomial,
    getGeneratorState,
    setGeneratorState,
    getPureGeneratorState,
    setPureGeneratorState,
  )
where

import Control.Concurrent
import Control.Concurrent.STM
import Control.Monad.IO.Class
import Control.Monad.STM
import Data.Int
import Data.Word
import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.Device
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Tensor
import Torch.TensorOptions

instance Show (TVar (Either (Word64, Device) (ForeignPtr ATen.Generator))) where
  show _ = "_"

newtype Generator = UnsafeGenerator
  { unGenerator :: TVar (Either (Word64, Device) (ForeignPtr ATen.Generator))
  }
  deriving (Eq, Show)

newGeneratorPtr :: Device -> Word64 -> IO (ForeignPtr ATen.Generator)
newGeneratorPtr device seed =
  case device of
    Device CPU _ -> ATen.newCPUGenerator seed
    Device CUDA idx -> do
      genPtr <- ATen.newCUDAGenerator (fromIntegral idx)
      ATen.generator_set_current_seed genPtr seed
      return genPtr
    Device MPS _ -> do
      genPtr <- ATen.newMPSGenerator
      ATen.generator_set_current_seed genPtr seed
      return genPtr

mkGenerator :: Device -> Word64 -> IO Generator
mkGenerator device seed = do
  genPtr <- newGeneratorPtr device seed
  UnsafeGenerator <$> newTVarIO (Right genPtr)

type RandomGenFunc = ForeignPtr ATen.IntArray -> ForeignPtr ATen.Generator -> ForeignPtr ATen.TensorOptions -> IO (ForeignPtr ATen.Tensor)

generatorFactory :: RandomGenFunc -> [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
generatorFactory func size options (UnsafeGenerator generator) =
  unsafePerformIO $ do
    mGenerator <- atomically $ do
      v <- readTVar generator
      case v of
        Right v' -> do
          let device =
                if generatorIsCuda v'
                  then Device {deviceType = CUDA, deviceIndex = fromIntegral $ generatorDevice v'}
                  else
                    if generatorIsMps v'
                    then Device {deviceType = MPS, deviceIndex = 0}
                    else Device {deviceType = CPU, deviceIndex = 0}
              seed = generatorSeed v'
          writeTVar generator $ seed `seq` deviceType device `seq` deviceIndex device `seq` Left (seed, device)
          return $ Right v'
        Left v -> return (Left v)
    genPtr <- case mGenerator of
      Right gen -> return gen
      Left (seed, device) -> newGeneratorPtr device seed
    tensor <- cast3 func size genPtr options
    nextGenenerator <- newTVarIO (Right genPtr)
    return (tensor, UnsafeGenerator nextGenenerator)
  where
    generatorIsCpu :: ForeignPtr ATen.Generator -> Bool
    generatorIsCpu gen = unsafePerformIO $ cast1 ATen.generator_is_cpu gen

    generatorIsCuda :: ForeignPtr ATen.Generator -> Bool
    generatorIsCuda gen = unsafePerformIO $ cast1 ATen.generator_is_cuda gen

    generatorIsMps :: ForeignPtr ATen.Generator -> Bool
    generatorIsMps gen = unsafePerformIO $ cast1 ATen.generator_is_mps gen

    generatorDevice :: ForeignPtr ATen.Generator -> Int
    generatorDevice gen = unsafePerformIO $ cast1 ATen.generator_get_device gen

    generatorSeed :: ForeignPtr ATen.Generator -> Word64
    generatorSeed gen = unsafePerformIO $ cast1 ATen.generator_current_seed gen

randn ::
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randn = generatorFactory LibTorch.randn_lGo

randn' ::
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randn' size = randn size defaultOpts

rand ::
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
rand = generatorFactory LibTorch.rand_lGo

rand' ::
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
rand' size = rand size defaultOpts

randint ::
  -- | low
  Int ->
  -- | high
  Int ->
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randint low high = generatorFactory (LibTorch.randint_lllGo (fromIntegral low) (fromIntegral high))

randint' ::
  -- | low
  Int ->
  -- | high
  Int ->
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randint' low high size = randint low high size defaultOpts

normal ::
  -- | mean
  Double ->
  -- | std
  Double ->
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
normal mean std = generatorFactory (LibTorch.normal_ddlGo (realToFrac mean) (realToFrac std))

normal' ::
  -- | mean
  Double ->
  -- | std
  Double ->
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
normal' mean std size = normal mean std size defaultOpts

-- | An RNG generator threaded as an immutable value: never mutated in place,
-- advanced by cloning on each draw (see 'withPureGenerator').
newtype PureGenerator = UnsafePureGenerator (ForeignPtr ATen.Generator)

-- | The wrapped pointer is an allocation address, not a description of the RNG
-- stream, so rendering it would make output vary between runs of the same
-- program. 'Generator' elides its contents for the same reason.
instance Show PureGenerator where
  show _ = "PureGenerator"

mkPureGenerator :: Device -> Word64 -> IO PureGenerator
mkPureGenerator device seed = UnsafePureGenerator <$> newGeneratorPtr device seed

-- | Draw from a 'PureGenerator', returning the result together with the
-- generator to use for the next draw. The argument is left untouched: ATen
-- advances a generator in place, so the draw runs against a clone.
withPureGenerator :: PureGenerator -> (ForeignPtr ATen.Generator -> IO a) -> (a, PureGenerator)
withPureGenerator (UnsafePureGenerator g) action = unsafePerformIO $ do
  g' <- ATen.generator_clone g
  a <- action g'
  pure (a, UnsafePureGenerator g')
{-# NOINLINE withPureGenerator #-}

multinomial ::
  -- | t
  Tensor ->
  -- | num_samples
  Int ->
  -- | replacement
  Bool ->
  -- | generator
  PureGenerator ->
  -- | output
  (Tensor, PureGenerator)
multinomial probs numSamples replacement gen =
  withPureGenerator gen (\g -> cast4 ATen.multinomial_tlbG probs numSamples replacement g)

-- | Snapshot the full RNG state (seed + stream offset) of a 'PureGenerator' as a
-- CPU ByteTensor. Restoring it with 'setPureGeneratorState' resumes the exact
-- stream position, unlike re-seeding which only preserves the seed. Works for
-- CPU (MT19937) and CUDA (Philox) generators.
getPureGeneratorState :: PureGenerator -> IO Tensor
getPureGeneratorState (UnsafePureGenerator g) = cast1 ATen.generator_get_state g

-- | Build a 'PureGenerator' resuming from a state captured by
-- 'getPureGeneratorState'. The source generator is left untouched (the state is
-- applied to a clone, matching the immutable-threading discipline).
setPureGeneratorState :: PureGenerator -> Tensor -> IO PureGenerator
setPureGeneratorState (UnsafePureGenerator g) st = do
  g' <- ATen.generator_clone g
  (cast2 ATen.generator_set_state g' st :: IO ())
  pure (UnsafePureGenerator g')

-- | Snapshot the full RNG state of a threaded 'Generator' as a CPU ByteTensor,
-- materializing the generator on its recorded device if it was de-materialized.
getGeneratorState :: Generator -> IO Tensor
getGeneratorState (UnsafeGenerator ref) = do
  v <- atomically $ readTVar ref
  g <- case v of
    Right g'        -> pure g'
    Left (seed, dv) -> newGeneratorPtr dv seed
  cast1 ATen.generator_get_state g

-- | Restore a 'Generator' from a state captured by 'getGeneratorState', yielding
-- a generator that resumes the exact stream position.
setGeneratorState :: Generator -> Tensor -> IO Generator
setGeneratorState (UnsafeGenerator ref) st = do
  v <- atomically $ readTVar ref
  g <- case v of
    Right g'        -> ATen.generator_clone g'
    Left (seed, dv) -> newGeneratorPtr dv seed
  (cast2 ATen.generator_set_state g st :: IO ())
  UnsafeGenerator <$> newTVarIO (Right g)
