{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

-- In this example we draw the gradient field of a given function f.

module Main where

import Graphics.Rendering.Chart.Backend.Cairo
import Graphics.Rendering.Chart.Easy hiding (makeAxis)
import Torch.Autograd (grad, makeIndependent, toDependent)
import qualified Torch.Functional as F
import Torch.NN
import Torch.Tensor (Tensor, asTensor, toDouble)

f :: Tensor -> Tensor -> Tensor
f x y = F.sin (2 * pit * r) where
    pit = asTensor (pi :: Double)
    r = F.sqrt (x * x + y * y)

makeAxis :: [Double] -> [Double] -> [(Double, Double)]
makeAxis axis1 axis2 = [(t, t') | t <- axis1, t' <- axis2]

computeGd :: (Double, Double) -> IO (Double, Double)
computeGd (x, y) = do
  tx <- makeIndependent (asTensor x)
  ty <- makeIndependent (asTensor y)
  let fxy = f (toDependent tx) (toDependent ty)
      gd = grad fxy [tx, ty]
      gdx = toDouble $ gd !! 0
      gdy = toDouble $ gd !! 1
  return (gdx, gdy)

main :: IO ()
main = do
  let n = 10
      xs = (\x -> x / n) <$> [(- 5 * n :: Double) .. 5 * n]
      ys = (\x -> x / n) <$> [(- 5 * n :: Double) .. 5 * n]
      grid = makeAxis xs ys
      fileName = "gd-field.png"
  vectors <- mapM computeGd grid
  toFile (FileOptions (1500,1500) PNG) fileName $ do
    setColors [opaque black]
    plot $ vectorField grid vectors
  putStrLn $ "\nCheck out " ++ fileName ++ "\n"
  where
    vectorField grid vectors = fmap plotVectorField $ liftEC $ do
      c <- takeColor
      plot_vectors_values .= zip grid vectors
      plot_vectors_style . vector_line_style . line_width .= 1
      plot_vectors_style . vector_line_style . line_color .= c
      plot_vectors_style . vector_head_style . point_radius .= 0.0
