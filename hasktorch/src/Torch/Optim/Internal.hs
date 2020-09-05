{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Optim.Internal where

import Foreign.ForeignPtr

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..), CppObject(..))
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.Optim as LibTorch

import Torch.Tensor
import Torch.Autograd
import Torch.NN

import Data.Default.Class

class OptimizerOption option optimizer | optimizer -> option where
  -- steps :: Parameterized d => option -> d -> (d -> IO Tensor) -> Int -> IO d
  initOptimizer :: Parameterized d => option -> d -> IO (optimizer d)
  step :: Parameterized d => optimizer d -> (d -> IO Tensor) -> IO Tensor
  -- Returned d depends on the state of optimizer.
  -- Do not call step function after this function is called.
  getParams :: Parameterized d => optimizer d -> IO d

--  next :: Parameterized d => optimizer d -> IO d

-- data AdagradOptions = AdagradOptions
--   { adagradLr :: Double
--   , adagradLrDecay :: Double
--   , adagradWeightDecay :: Double
--   , adagradInitialAccumulatorValue :: Double
--   , adagradEps :: Double
--   } deriving (Show, Eq)

data AdamOptions = AdamOptions
  { adamLr :: Double
  , adamBetas :: (Double,Double)
  , adamEps :: Double
  , adamWeightDecay :: Double
  , adamAmsgrad :: Bool
  } deriving (Show, Eq)

data Adam p = Adam (ForeignPtr ATen.Adam) p

instance Default AdamOptions where
  def = AdamOptions
    { adamLr = 1e-3
    , adamBetas = (0.9, 0.999)
    , adamEps = 1e-8
    , adamWeightDecay = 0
    , adamAmsgrad = False
    } 

optimizerWithAdam'
  :: AdamOptions
  -> [Parameter]
  -> ([Parameter] -> IO Tensor)
  -> Int
  -> IO [Parameter]
optimizerWithAdam' AdamOptions{..} initParams loss numIter = do
  v <- cast9 LibTorch.optimizerWithAdam adamLr (fst adamBetas) (snd adamBetas) adamEps adamWeightDecay adamAmsgrad initParams' (trans loss) numIter
  return $ map IndependentTensor v
  where
    initParams' = map toDependent initParams
    trans :: ([Parameter] -> IO Tensor) -> ForeignPtr ATen.TensorList -> IO (ForeignPtr ATen.Tensor)
    trans func inputs =
      uncast inputs $ \inputs' -> do
        (Unsafe ret) <- func $ map (IndependentTensor . Unsafe) inputs'
        cast ret return

adam
  :: Parameterized d
  => AdamOptions
  -> d
  -> IO (Adam d)
adam AdamOptions{..} initParams = do
  v <- cast7 LibTorch.adam adamLr (fst adamBetas) (snd adamBetas) adamEps adamWeightDecay adamAmsgrad initParams'
  return $ Adam v initParams
  where
    initParams' = map toDependent $ flattenParameters initParams

stepAdam
  :: Parameterized d
  => Adam d
  -> (d -> IO Tensor)
  -> IO Tensor
stepAdam (Adam optimizer initParams) loss = cast0 (LibTorch.stepAdam optimizer trans)
  where
    trans :: ForeignPtr ATen.TensorList -> IO (ForeignPtr ATen.Tensor)
    trans inputs =
      uncast inputs $ \inputs' -> do
        (Unsafe ret) <- loss $ replaceParameters initParams $  map (IndependentTensor . Unsafe) inputs'
        cast ret return

instance OptimizerOption AdamOptions Adam where
  -- steps opt initParams loss numIter = do
  --   v <- optimizerWithAdam' opt (flattenParameters initParams) (\params -> loss (replaceParameters initParams params))  numIter
  --   return $ replaceParameters initParams v
  initOptimizer = adam
  step = stepAdam
  getParams (Adam optimizer initParams) = fmap (replaceParameters initParams . map (IndependentTensor . Unsafe)) $ cast0 (LibTorch.getAdamParams optimizer)

-- data AdamwOptions = AdamwOptions
--   { adamwLr :: Double
--   , adamwBetas :: (Double,Double)
--   , adamwEps :: Double
--   , adamwWeightDecay :: Double
--   , adamwAmsgrad :: Bool
--   } deriving (Show, Eq)

-- data LbfgsOptions = LbfgsOptions
--   { lbfgsLr :: Double
--   , lbfgsMaxIter :: Int
--   , lbfgsMaxEval :: Maybe Int
--   , lbfgsToleranceGrad :: Double
--   , lbfgsToleranceChange :: Double
--   , lbfgsHistorySize :: Int
--   , lbfgsLineSearchFn :: String
--   } deriving (Show, Eq)

-- data RmspropOptions = RmspropOptions
--   { rmspropLr :: Double
--   , rmspropBetas :: (Double,Double)
--   , rmspropEps :: Double
--   , rmspropWeightDecay :: Double
--   , rmspropAmsgrad :: Bool
--   } deriving (Show, Eq)

-- data SgdOptions = SgdOptions
--   { sgdLr :: Double
--   , sgdMomentum :: Double
--   , sgdDampening :: Double
--   , sgdWeightDecay :: Double
--   , sgdNesterov :: Bool
--   } deriving (Show, Eq)
