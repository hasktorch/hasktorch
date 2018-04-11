{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# OPTIONS_GHC -Wno-type-defaults -Wno-unused-local-binds -fno-cse #-}
module Main where

import Data.Monoid
import GHC.TypeLits
import Lens.Micro ((^.), _2, _1)
import System.IO.Unsafe

import Torch hiding (N, abs, sqrt, round, max)
import qualified Torch.Core.Random as RNG
import qualified Torch as T
import qualified Torch as TU

type N = 20 -- sample size
type M = 2
type Tensor = FloatTensor
type Precision = Float
type AccPrecision = Double
type Epsilon = Precision

seedVal :: RNG.Seed
seedVal = 3141592653579

genData :: Tensor '[M, 1] -> IO (Tensor '[N, M], Tensor '[N, 1])
genData w = do
  gen <- RNG.new
  RNG.manualSeed gen seedVal
  noise        :: Tensor '[N, 1] <- T.normal gen 0 0.10
  predictorVal :: Tensor '[N]    <- T.normal gen 0 1.0
  ones         :: Tensor '[N] <- T.constant 1
  x :: Tensor '[N, M] <- (predictorVal `T.cat1d` ones) >>= T.resizeAs
  y :: Tensor '[N, 1] <- T.resizeAs ((x !*! w) ^+^ noise)
  pure (x, y)

loss :: (Tensor '[N, M], Tensor '[N, 1]) -> Tensor '[M, 1] -> Precision
loss (x, y) w = squaredSum (y ^-^ (x !*! w)) + l1 w
 where
  squaredSum :: Tensor '[N, 1] -> Precision
  squaredSum t = unsafePerformIO $ fmap realToFrac . TU.sumall =<< TU.square t

l1 :: (Fractional prec, Real prec) => Tensor '[M, 1] -> prec
l1 t = unsafePerformIO $ fmap realToFrac . TU.sumall =<< TU.abs t
{-# NOINLINE l1 #-}

l2 :: (Fractional prec, Real prec) => Tensor '[M, 1] -> prec
l2 t = unsafePerformIO $ fmap (realToFrac . sqrt) . T.sumall =<< T.square t
{-# NOINLINE l2 #-}

prox_l1 :: Tensor '[M, 1] -> Precision -> IO (Tensor '[M, 1])
prox_l1 w l = do
  a <- T.sign w
  b <- max_plus (w ^- l)
  pure (a * b)
 where
   max_plus :: Tensor '[M, 1] -> IO (Tensor '[M, 1])
   max_plus t = T.zerosLike >>= \z -> T.cmax t z

prox_l1_single :: AccPrecision -> AccPrecision -> Precision
prox_l1_single w_i l = realToFrac (signum w_i * max (w_i - l) 0)

data LassoDim = D1 | D2
  deriving (Enum, Ord, Show, Eq)

-- TODO: move Int-based version into hasktorch-dimensions
getDim2d
  :: (Fractional precision, Real precision)
  => Tensor '[N, M] -> LassoDim -> precision
getDim2d _ = \case
  D1 -> realToFrac (natVal (Proxy :: Proxy N))
  D2 -> realToFrac (natVal (Proxy :: Proxy M))

coordinate_descent
  :: (Tensor '[N, M], Tensor '[N, 1])
  -> Precision
  -> Integer
  -> Tensor '[M, 1]
  -> IO [(Tensor '[M, 1], Precision)]
coordinate_descent (x, y) l = go 0 []
 where
  nSamples = getDim2d x D1

  nParams :: Int
  nParams = round ( getDim2d x D2 )

  go :: Int -> [(Tensor '[M, 1], Precision)] -> Integer -> Tensor '[M, 1] -> IO [(Tensor '[M, 1], Precision)]
  go ix res j w
    | j > natVal (Proxy :: Proxy M) - 1 = pure ((w, loss (x, y) w):res)
    | otherwise = do
      let jIdx = fromIntegral j
      x_j <- T.getColumn x jIdx
      w_j <- T.getRow w jIdx

      let r_j     = (y ^-^ (x !*! w)) ^+^ (x_j !*! w_j)
      let w_j_upd = prox_l1_single ((1 / nSamples) * (x_j <.> r_j)) (realToFrac l)
      w_upd <- T.copy w
      T.setElem2d w_upd jIdx 0 w_j_upd
      let loss_obj = loss (x, y) w_upd

      go (ix+1) ((w_upd, loss_obj):res) (j + 1) w_upd

cyclic_coordinate_descent
  :: (Tensor '[N, M], Tensor '[N, 1])
  -> Precision
  -> Epsilon
  -> Tensor '[M, 1]
  -> IO [(Tensor '[M, 1], Precision)]
cyclic_coordinate_descent (x, y) l eps = go []
 where
  nCoords :: Int
  nCoords = round ( getDim2d x D2 )

  go :: [(Tensor '[M, 1], Precision)] -> Tensor '[M, 1] -> IO [(Tensor '[M, 1], Precision)]
  go res w = do
    iter_coord <- coordinate_descent (x, y) l 0 w
    let w_upd = (^. _1) . last . take nCoords $ iter_coord
    let loss_w     = loss (x, y) w
    let loss_w_upd = loss (x, y) w_upd
    if abs (loss_w - loss_w_upd) < eps
    then pure $ (w, loss_w):res
    else go ([(w, loss_w)] <> iter_coord <> res) w_upd

run_cd_synthetic :: Int -> Precision -> IO (Tensor '[M, 1])
run_cd_synthetic iters l = do
  gen       <- RNG.new
  trueParam <- T.normal gen 20 1
  dat       <- genData trueParam

  -- Setup CD
  p :: Tensor '[M, 1] <- T.zerosLike
  lzy <- take iters <$> cyclic_coordinate_descent dat l 0.0001 p
  print (fmap snd lzy)

  let final    = last lzy
      w        = (^. _1) final
      obj      = (^. _2) final
      accuracy = abs $ (snd . last $ lzy) - (snd . last . init $ lzy)

  putStrLn $ "Loss " <> show obj <> " accuracy of " <> show accuracy
  pure w

run_fista_synthetic :: Int -> Precision -> IO (Tensor '[M, 1])
run_fista_synthetic iters l = do
  gen       <- RNG.new
  trueParam <- T.normal gen 20.0 1.0
  dat       <- genData trueParam
  w0 :: Tensor '[M, 1] <- T.zerosLike
  z0 :: Tensor '[M, 1] <- T.zerosLike
  lzy <- fmap (take iters) $ fista dat l 1 0.0001 w0 z0 1

  -- Setup CD
  let final    = last lzy
      w        = (^. _1) final
      obj      = (^. _2) final
      accuracy = abs $ (snd . last $ lzy) - (snd . head . tail . reverse $ lzy)
  putStrLn $ "Loss " <> show obj <> " accuracy of " <> show accuracy
  pure w

backtracking
  :: (Tensor '[N, M], Tensor '[N, 1])
  -> Tensor '[M, 1]
  -> Tensor '[M, 1]
  -> Tensor '[M, 1]
  -> Precision
  -> Precision
  -> IO Precision
backtracking (x, y) w z g l est_L = do
  w_next <- prox_l1 (z ^-^ (rate *^ g)) (rate * l)
  if loss (x, y) w > q
  then backtracking (x, y) w_next z g l (est_L * eta)
  else pure est_L
 where
  dst      = w ^-^ z
  fz       = loss (x, y) z - l1 z
  q        = fz + realToFrac (dst  <.> g) + (est_L / 2) * realToFrac (dst  <.> dst ) + l * l1 w
  nSamples = getDim2d x D1
  rate     = 1.0 / est_L
  eta      = 1.5


fista
  :: (Tensor '[N, M], Tensor '[N, 1])
  -> Precision
  -> Precision
  -> Epsilon
  -> Tensor '[M, 1]
  -> Tensor '[M, 1]
  -> Precision
  -> IO [(Tensor '[M, 1], Precision)]
fista (x, y) l est_L eps = go []
 where
  go :: [(Tensor '[M, 1], Precision)] -> Tensor '[M, 1] -> Tensor '[M, 1] -> Precision -> IO [(Tensor '[M, 1], Precision)]
  go res w z t = do
    let rate_k = 1.0 / est_L                                  -- initial estimation
    g         <- gradient (x, y) z                            -- current gradient
    w_k       <- prox_l1 (z ^-^ (rate_k *^ g)) ( rate_k * l ) -- initial proximal step
    lipschitz <- backtracking (x, y) w_k z g l est_L          -- backtracking from first proximal step
    let rate   = 1 / lipschitz
    w_next    <- prox_l1 (z - rate *^ g) ( rate * l )         -- good proximal step with estimated L

    let
      t_next      = ( 1.0 + sqrt (1 + 4 * t ^ 2) ) / 2.0
      z_next      = w_next ^+^ ( ((t - 1.0) / t_next) *^ (w_next ^-^ w) )
      loss_w      = loss (x, y) w
      loss_w_next = loss (x, y) w_next
    if abs (loss_w_next - loss_w) < eps
    then pure res
    else go ((w_next, loss_w_next):res) w_next z_next t_next

gradient
  :: (Tensor '[N, M], Tensor '[N, 1])
  -> Tensor '[M, 1]
  -> IO (Tensor '[M, 1])
gradient (x, y) w = do
  xT <- T.newTranspose2d x
  pure $ (1.0 / nSamples) *^ xT !*! (x !*! w - y)
 where
  nSamples = getDim2d x D1

main :: IO ()
main = do
  putStrLn "\nRun using the same random seed"
  _ <- run_cd_synthetic 100 1.0
  _ <- run_fista_synthetic 100 1.0
  pure ()
