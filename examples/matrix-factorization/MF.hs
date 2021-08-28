{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

-- In matrix factorization, we aim to write a given matrix R (of size n x m) as
-- a product U * V where U is n x k and V is k x m for some k. This is done in
-- this example by minimizing a loss function which is the mean square of the
-- difference between U * V and R.

module MF where

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (matmul)
import Torch.NN
  ( Parameter,
    Randomizable,
    sample,
  )
import Torch.Tensor (Tensor)
import Torch.TensorFactories (randnIO')

data MatrixFactSpec = MatrixFactSpec
  { dim1 :: Int, -- that is n when R has n x m shape
    dim2 :: Int, -- that is m when R has n x m shape
    common_dim :: Int -- that is k when U has n x k shape
  }
  deriving (Show, Eq)

data MatrixFact = MatrixFact {u :: Parameter, v :: Parameter}

instance Randomizable MatrixFactSpec MatrixFact where
  sample (MatrixFactSpec n m k) = do
    u <- makeIndependent =<< randnIO' [n, k]
    v <- makeIndependent =<< randnIO' [k, m]
    pure $ MatrixFact u v

flattenParameters :: MatrixFact -> [Parameter]
flattenParameters (MatrixFact u v) = [u, v]

replaceParameters :: MatrixFact -> [Parameter] -> MatrixFact
replaceParameters state params_list = MatrixFact u v
  where
    u = params_list !! 0
    v = params_list !! 1

mulMF :: MatrixFact -> Tensor
mulMF (MatrixFact u v) = matmul u' v'
  where
    u' = toDependent u
    v' = toDependent v
