{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module OptimSpec where

import Control.Monad (when)
import GHC.Generics
import Test.Hspec
import Text.Printf (printf)
import Torch.Autograd
import Torch.Functional
import Torch.NN
import Torch.Optim
import Torch.Tensor
import Torch.TensorFactories (eye', ones', randIO', randnIO', zeros')
import Prelude hiding (cos, exp, sqrt)
import qualified Prelude as P

-- Convex Quadratic

data ConvQuadSpec = ConvQuadSpec {n :: Int}

data ConvQuad = ConvQuad {w :: Parameter} deriving (Show, Generic)

instance Randomizable ConvQuadSpec ConvQuad where
  sample (ConvQuadSpec n) = do
    w <- makeIndependent =<< randnIO' [n]
    pure $ ConvQuad w

instance Parameterized ConvQuad

convexQuadratic :: Tensor -> Tensor -> Tensor -> Tensor
convexQuadratic a b w =
  mulScalar (0.5 :: Float) (dot (mv a w) w) - dot w b

lossConvQuad :: Tensor -> Tensor -> ConvQuad -> Tensor
lossConvQuad a b (ConvQuad w) = convexQuadratic a b w'
  where
    w' = toDependent w

-- 2D Rosenbrock

data RosenSpec = RosenSpec deriving (Show, Eq)

data Rosen = Rosen {x :: Parameter, y :: Parameter} deriving (Generic)

instance Show Rosen where
  show (Rosen x y) = show (extract x :: Float, extract y :: Float)
    where
      extract :: TensorLike a => Parameter -> a
      extract p = asValue $ toDependent p

instance Randomizable RosenSpec Rosen where
  sample RosenSpec = do
    x <- makeIndependent =<< randnIO' [1]
    y <- makeIndependent =<< randnIO' [1]
    pure $ Rosen x y

-- instance Parameterized Rosen

instance Parameterized Rosen where
  -- flattenParameters :: f -> [Parameter]
  flattenParameters (Rosen x y) = [x, y]

rosenbrock2d :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock2d a b x y = square (addScalar a $ (-1.0) * x) + mulScalar b (square (y - x * x))
  where
    square = pow (2 :: Int)

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: Rosen -> Tensor
lossRosen Rosen {..} = rosenbrock' (toDependent x) (toDependent y)

-- Ackley function

data AckleySpec = AckleySpec deriving (Show, Eq)

data Ackley = Ackley {pos :: Parameter} deriving (Show, Generic)

instance Randomizable AckleySpec Ackley where
  sample AckleySpec = do
    pos <- makeIndependent =<< randnIO' [2]
    pure $ Ackley pos

instance Parameterized Ackley

ackley :: Float -> Float -> Float -> Tensor -> Tensor
ackley a b c x =
  mulScalar (- a) (exp (- b' * (sqrt $ (sumAll (x * x)) / d)))
    - exp (1.0 / d * sumAll (cos (mulScalar c x)))
    + (asTensor $ a + P.exp 1.0)
  where
    b' = asTensor b
    c' = asTensor c
    d = asTensor . product $ shape x

ackley' = ackley 20.0 0.2 (2 * pi :: Float)

lossAckley :: Ackley -> Tensor
lossAckley (Ackley x) = ackley' x'
  where
    x' = toDependent x

-- | show output after n iterations (not used for tests)
showLog :: (Show a) => Int -> Int -> Int -> Tensor -> a -> IO ()
showLog n i maxIter lossValue state =
  when (i == 0 || mod i n == 0 || i == maxIter -1) $ do
    putStrLn
      ( "Iter: " ++ printf "%6d" i
          ++ " | Loss:"
          ++ printf "%05.4f" (asValue lossValue :: Float)
          ++ " | Parameters: "
          ++ show state
      )

-- | Optimize convex quadratic with specified optimizer
optConvQuad :: (Optimizer o) => Int -> o -> IO ()
optConvQuad numIter optInit = do
  let dim = 2
      a = eye' dim dim
      b = zeros' [dim]
  paramInit <- sample $ ConvQuadSpec dim
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = (lossConvQuad a b) paramState
    runStep paramState optState lossValue 5e-4
  pure ()

-- | Optimize Rosenbrock function with specified optimizer
optRosen :: (Optimizer o) => Int -> o -> IO ()
optRosen numIter optInit = do
  paramInit <- sample RosenSpec
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = lossRosen paramState
    runStep paramState optState lossValue 5e-4
  pure ()

-- | Optimize Ackley function with specified optimizer
optAckley :: (Optimizer o) => Int -> o -> IO ()
optAckley numIter optInit = do
  paramInit <- sample AckleySpec
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = lossAckley paramState
    runStep paramState optState lossValue 5e-4
  pure ()

-- | Check global minimum point for Rosenbrock
checkGlobalMinRosen :: IO ()
checkGlobalMinRosen = do
  putStrLn "\nCheck Actual Global Minimum (at 1, 1):"
  print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

-- | Check global minimum point for Convex Quadratic
checkGlobalMinConvQuad :: IO ()
checkGlobalMinConvQuad = do
  putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
  let dim = 2
      a = eye' dim dim
      b = zeros' [dim]
  print $ convexQuadratic a b (zeros' [dim])

-- | Check global minimum point for Ackley
checkGlobalMinAckley :: IO ()
checkGlobalMinAckley = do
  putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
  print $ ackley' (zeros' [2])

main :: IO ()
main = do
  let numIter = 20000

  -- Convex Quadratic w/ GD, GD+Momentum, Adam
  putStrLn "\nConvex Quadratic\n================"
  putStrLn "\nGD"
  optConvQuad numIter GD
  putStrLn "\nGD + Momentum"
  optConvQuad numIter (GDM 0.9 [zeros' [2]])
  putStrLn "\nAdam"
  optConvQuad
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
  checkGlobalMinConvQuad

  -- 2D Rosenbrock w/ GD, GD+Momentum, Adam
  putStrLn "\n2D Rosenbrock\n================"
  putStrLn "\nGD"
  optRosen numIter GD
  putStrLn "\nGD + Momentum"
  optRosen numIter (GDM 0.9 [zeros' [1], zeros' [1]])
  putStrLn "\nAdam"
  optRosen
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
  checkGlobalMinRosen

  -- Ackley w/ GD, GD+Momentum, Adam
  putStrLn "\nAckley (Gradient methods fail)\n================"
  putStrLn "\nGD"
  optAckley numIter GD
  putStrLn "\nGD + Momentum"
  optAckley numIter (GDM 0.9 [zeros' [1], zeros' [1]])
  putStrLn "\nAdam"
  optAckley
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
  checkGlobalMinAckley

spec :: Spec
spec = do
  it "ConvQuad GD" $ do
    optConvQuad numIter GD
  it "ConvQuad GDM" $ do
    optConvQuad numIter (GDM 0.9 [zeros' [2]])
  it "ConvQuad Adam" $ do
    optConvQuad
      numIter
      Adam
        { beta1 = 0.9,
          beta2 = 0.999,
          m1 = [zeros' [1], zeros' [1]],
          m2 = [zeros' [1], zeros' [1]],
          iter = 0
        }
  it "Rosen GD" $ do
    optRosen numIter GD
  it "Rosen GDM" $ do
    optRosen numIter (GDM 0.9 [zeros' [1], zeros' [1]])
  it "Rosen Adam" $ do
    optRosen
      numIter
      Adam
        { beta1 = 0.9,
          beta2 = 0.999,
          m1 = [zeros' [1], zeros' [1]],
          m2 = [zeros' [1], zeros' [1]],
          iter = 0
        }
  it "Ackley GD" $ do
    optAckley numIter GD
  it "Ackley GDM" $ do
    optAckley numIter (GDM 0.9 [zeros' [1], zeros' [1]])
  it "Ackley Adam" $ do
    optAckley
      numIter
      Adam
        { beta1 = 0.9,
          beta2 = 0.999,
          m1 = [zeros' [1], zeros' [1]],
          m2 = [zeros' [1], zeros' [1]],
          iter = 0
        }
  where
    numIter = 100
