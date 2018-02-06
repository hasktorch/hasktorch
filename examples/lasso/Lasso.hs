{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

module Main where

import           Data.Function                         ((&))
import           Data.Monoid                           ((<>))
import           Data.Singletons
import           GHC.TypeLits
import           Lens.Micro

import           Torch.Core.Tensor.Dim                 hiding (N)
import           Torch.Core.Tensor.Static.Double
import           Torch.Core.Tensor.Static.DoubleMath
import           Torch.Core.Tensor.Static.DoubleRandom

-- type N = 2000 -- sample size
-- type M = 200
type N = 20 -- sample size
type M = 2

seedVal :: Int
seedVal = 3141592653579

genData :: TDS '[M, 1] -> IO (TDS '[N, M], TDS '[N, 1])
genData w = do
  gen <- newRNG
  manualSeed gen seedVal
  noise        :: TDS '[N, 1] <- tds_normal gen 0.0 0.10
  predictorVal :: TDS '[N] <- tds_normal gen 0.0 1.0
  let x :: TDS '[N, M] =
        predictorVal
        & tds_cat (tds_init 1.0)
        & tds_resize
      y = (x !*! w) ^+^ noise
          & tds_resize :: TDS '[N, 1]
  pure (x, y)

loss :: (TDS '[N, M], TDS '[N, 1]) -> TDS '[M, 1] -> Double
loss (x, y) w =
  ( tds_sumAll . tds_square $ y ^-^ (x !*! w) ) + l1 w

l1 :: TDS '[M, 1] -> Double
l1 = tds_sumAll . tds_abs

l2 :: TDS '[M, 1] -> Double
l2 = tds_sumAll . tds_square

prox_l1 :: TDS '[M, 1] -> Double -> TDS '[M, 1]
prox_l1 w l        = tds_sign w ^*^ max_plus (w ^- l)
  where max_plus t = tds_cmax t (tds_new :: TDS '[M, 1])

prox_l1_single :: Double -> Double -> Double
prox_l1_single w_i l = signum w_i * max (w_i - l) 0.0

tds_getDim2d :: TDS '[N, M] -> Int -> Double
tds_getDim2d _ i
  | i == 0 = ((realToFrac $ natVal (Proxy :: Proxy N)) :: Double)
  | i == 1 = ((realToFrac $ natVal (Proxy :: Proxy M)) :: Double)
  | otherwise = error "Only works on 2d tensors; `i` must be either 0 or 1"

coordinate_descent ::
  (TDS '[N, M], TDS '[N, 1])
 -> TDS '[M, 1]
 -> Double
 -> Int
 -> [(TDS '[M, 1], Double)]
coordinate_descent (x, y) w l j
  | j > (round ( tds_getDim2d x 1 ) :: Int) - 1 =
      []
  | otherwise    = [(w_upd, loss_obj)] <> coordinate_descent (x, y) w_upd l (j + 1)
  where nSamples = tds_getDim2d x 0
        nParams  = round ( tds_getDim2d x 1 ) :: Int
        x_j      = tds_getColumn x ( toInteger j )
        w_j      = tds_getRow w (toInteger j)
        r_j      = y - x !*! w + x_j !*! w_j
        w_j_upd  = prox_l1_single ((1.0 / nSamples) * (x_j <.> r_j)) l
        w_upd    = tds_setElem w j 0 w_j_upd
        loss_obj = loss (x, y) w_upd

cyclic_coordinate_descent ::
  (TDS '[N, M], TDS '[N, 1])
  -> TDS '[M, 1]
  -> Double
  -> Double
  -> [(TDS '[M, 1], Double)]
cyclic_coordinate_descent (x, y) w l eps =
  if abs (loss (x, y) w - loss (x, y) w_upd) < eps then
    []
  else
    [(w, loss_obj)] <> iter_coord <> cyclic_coordinate_descent (x, y) w_upd l eps
  where nCoords    = round ( tds_getDim2d x 1 ) :: Int
        iter_coord = take nCoords (coordinate_descent (x, y) w l 0)
        w_upd      = (^. _1) . last $ iter_coord
        loss_obj   = loss (x, y) w

run_cd_synthetic :: Int -> Double -> IO (TDS '[M, 1])
run_cd_synthetic iters l = do
  gen       <- newRNG
  trueParam <- tds_normal gen 20.0 1.0
  dat       <- genData trueParam

  -- Setup CD
  let p                 = tds_new :: TDS '[M, 1]
      lazy              = take iters $ cyclic_coordinate_descent dat p l 0.0001
      final             = last lazy
      w                 = (^. _1) final
      obj               = (^. _2) final
      accuracy          = abs $ (snd . last $ lazy) - (snd . head . tail . reverse $ lazy)
  putStrLn $ "Loss " <> show obj <> " accuracy of " <> show accuracy
  pure w

run_fista_synthetic :: Int -> Double -> IO (TDS '[M, 1])
run_fista_synthetic iters l = do
  gen       <- newRNG
  trueParam <- tds_normal gen 20.0 1.0
  dat       <- genData trueParam

  -- Setup CD
  let w0                = tds_new :: TDS '[M, 1]
      z0                = tds_new :: TDS '[M, 1]
      lazy              = take iters $ fista dat w0 z0 l 1.0 1.0 0.0001
      final             = last lazy
      w                 = (^. _1) final
      obj               = (^. _2) final
      accuracy          = abs $ (snd . last $ lazy) - (snd . head . tail . reverse $ lazy)
  putStrLn $ "Loss " <> show obj <> " accuracy of " <> show accuracy
  pure w

gradient ::
  (TDS '[N, M], TDS '[N, 1])
  -> TDS '[M, 1]
  -> TDS '[M, 1]
gradient (x, y) w =
  (1.0 / nSamples) *^ xT !*! (x !*! w - y)
  where xT       = tds_trans x
        nSamples = tds_getDim2d x 0

backtracking ::
  (TDS '[N, M], TDS '[N, 1])
  -> TDS '[M, 1]
  -> TDS '[M, 1]
  -> TDS '[M, 1]
  -> Double
  -> Double
  -> Double
backtracking (x, y) w z g l est_L =
  if loss (x, y) w > q then
    backtracking (x, y) w_next z g l (est_L * eta)
  else
    est_L
  where w_next   = prox_l1 (z - rate *^ g) ( rate * l )
        fz       = loss (x, y) z - l1 z
        dist     = w ^-^ z
        q        = fz + dist <.> g + (est_L / 2.0) * dist <.> dist + l * l1 w
        nSamples = tds_getDim2d x 0
        rate     = 1.0 / est_L
        eta      = 1.5

fista ::
  (TDS '[N, M], TDS '[N, 1])
  -> TDS '[M, 1]
  -> TDS '[M, 1]
  -> Double
  -> Double
  -> Double
  -> Double
  -> [(TDS '[M, 1], Double)]
fista (x, y) w z l t est_L eps =
  if ( abs $ loss (x, y) w_next - loss (x, y) w ) < eps then
     [(w_next, loss (x, y) w_next)]
  else
    [(w_next, loss (x, y) w_next)] <> fista (x, y) w_next z_next l t_next est_L eps
  where w_next    = prox_l1 (z - rate *^ g) ( rate * l ) -- good proximal step with estimated L
        w_k       = prox_l1 (z - rate_k *^ g) ( rate_k * l ) -- initial proximal step
        g         = gradient (x, y) z
        rate      = 1.0 / lipschitz
        rate_k    = 1.0 / est_L -- initial estimation
        lipschitz = backtracking (x, y) w_k z g l est_L -- backtracking from first proximal step
        t_next    = ( 1.0 + sqrt (1 + 4 * t ^ 2) ) / 2.0
        z_next    = w_next ^+^ ( ((t - 1.0) / t_next) *^ (w_next ^-^ w) )

main :: IO ()
main = do
  putStrLn "\nRun using the same random seed"
  _ <- run_cd_synthetic 100 1.0
  _ <- run_fista_synthetic 100 1.0
  pure ()
