{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module MLP where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters
import LinearLayer


data MLPSpec = MLPSpec { feature_counts :: [Int], nonlinearitySpec :: Tensor -> Tensor }

data MLP = MLP { layers :: [Linear], nonlinearity :: Tensor -> Tensor }

instance Randomizable MLPSpec MLP where
  sample MLPSpec{..} = do
      let layer_sizes = mkLayerSizes feature_counts
      linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
      return $ MLP { layers = linears, nonlinearity = nonlinearitySpec }
    where
      mkLayerSizes (a : (b : t)) =
          scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

instance Parametrized MLP where
  flattenParameters MLP{..} = concat $ map flattenParameters layers
  replaceOwnParameters mlp = do
    new_layers <- mapM replaceOwnParameters (layers mlp)
    return $ mlp { layers = new_layers }

instance Show MLP where
  show (MLP layers _) = 
    "MLP with " ++ (show $ length layers) ++ " layers\n" ++
    "----------------------------------------------------\n" ++ 
    (join $ map showLayer layers)
    where
      showLayer :: Linear -> String
      showLayer linearLayer = 
        "Linear Layer\n" ++ 
        "Weights:\n" ++ (show $ weight linearLayer) ++ "\n" ++
        "Bias:\n" ++ (show $ bias linearLayer) ++ "\n" ++
        "--------------------------------------------------------\n"


mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where revApply x f = f x
