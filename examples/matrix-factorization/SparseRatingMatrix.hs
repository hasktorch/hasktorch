{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (foldM, when)
import MF
import Torch.Autograd (grad, makeIndependent, toDependent)
import Torch.DType (DType (Double))
import Torch.Functional (mseLoss, toDType)
import Torch.Functional.Internal (bmm)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.NN
  ( Randomizable,
    sample,
  )
import Torch.Optim (sgd)
import Torch.Tensor (Tensor, asTensor, indexSelect, reshape, toDouble, toInt, (!))
import Torch.TensorFactories (randintIO')

data RatingBatch = RatingBatch Items Users RatingValues deriving (Show)

type Items = [Int]

type Users = [Int]

type RatingValues = [Double]

data RatingsSpec = RatingsSpec {usersNumber :: Int, itemsNumber :: Int, ratingsNumber :: Int}

instance Randomizable RatingsSpec RatingBatch where
  sample (RatingsSpec n m l) = do
    items <- randintIO' 1 n [l]
    users <- randintIO' 1 m [l]
    ratings <- randintIO' 1 5 [l]
    let list_ratings = toDouble <$> ((!) ratings) <$> [0 .. (l -1)]
        list_users = toInt <$> ((!) users) <$> [0 .. (l -1)]
        list_items = toInt <$> ((!) items) <$> [0 .. (l -1)]
    pure $ RatingBatch list_items list_users list_ratings

lossMF :: MatrixFact -> RatingBatch -> Tensor
lossMF (MatrixFact u v) (RatingBatch items users ratings) = loss
  where
    batch_size = length ratings
    ratings_tensor = asTensor (ratings :: [Double])
    items_ids = (\x -> x -1) <$> items
    users_ids = (\x -> x -1) <$> users
    u' = toDependent u
    v' = toDependent v
    u'' = indexSelect (0 :: Int) (asTensor (items_ids :: [Int])) u'
    v'' = indexSelect (1 :: Int) (asTensor (users_ids :: [Int])) v'
    u''' = reshape [batch_size, 1, -1] u''
    v''' = reshape [batch_size, -1, 1] v''
    uv_ui = toDType (Double :: DType) $ bmm u''' v'''
    loss = mseLoss ratings_tensor uv_ui

n_users = 500

n_items = 500

n_ratings = 1000

batch_size = 100

num_iters = 2000

main :: IO ()
main = do
  manual_seed_L 123
  ratingBatch@(RatingBatch items users ratings) <- sample $ (RatingsSpec n_users n_items n_ratings)
  let k = 10
      foldLoop x count block = foldM block x [1 .. count]
      gdBlock state i = do
        random_start <- randintIO' 0 (n_ratings - batch_size) [1]
        let start = toInt $ random_start ! (0 :: Int)
            items_batch = (!!) items <$> [start .. start + batch_size -1]
            users_batch = (!!) users <$> [start .. start + batch_size -1]
            ratings_batch = (!!) ratings <$> [start .. start + batch_size -1]
            loss = lossMF state (RatingBatch items_batch users_batch ratings_batch)
            flat_parameters = flattenParameters state
            gradients = grad loss flat_parameters
        when (i `mod` 100 == 0) do
          putStrLn $
            "Iteration: " ++ show i
              ++ " | Mean Squared Loss: "
              ++ show ((toDouble $ lossMF state ratingBatch))
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        return $ replaceParameters state new_flat_parameters
  putStrLn $ "No. Users: " ++ (show n_users)
  putStrLn $ "No. Items: " ++ (show n_items)
  putStrLn $ "No. Ratings: " ++ (show n_ratings)
  init <- sample $ (MatrixFactSpec n_users n_items k)
  trained <- foldLoop init num_iters gdBlock
  pure ()
