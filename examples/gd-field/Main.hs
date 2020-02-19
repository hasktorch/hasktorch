{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

-- In this example we draw the heatmap of the norm of gradient field
-- of a given function f.

module Main where

import Data.Text (Text, pack)
import Graphics.Vega.VegaLite
import Torch.Autograd (grad, makeIndependent, toDependent)
import qualified Torch.Functional as F
import Torch.NN
import Torch.Tensor (Tensor, asTensor, toDouble)

f :: Tensor -> Tensor -> Tensor
f x y = F.sin (2 * pit * r)
  where
    pit = asTensor (pi :: Double)
    r = F.sqrt (x * x + y * y)

makeAxis :: [Double] -> [Double] -> [(Double, Double)]
makeAxis axis1 axis2 = [(t, t') | t <- axis1, t' <- axis2]

computeGd :: (Double, Double) -> IO (Double, Double, Double)
computeGd (x, y) = do
  tx <- makeIndependent (asTensor x)
  ty <- makeIndependent (asTensor y)
  let fxy = f (toDependent tx) (toDependent ty)
      gd = grad fxy [tx, ty]
      gdx = toDouble $ gd !! 0
      gdy = toDouble $ gd !! 1
      gdr = sqrt (gdx * gdx + gdy * gdy)
  return (x, y, gdr)

main :: IO ()
main = do
  let n = 30
      b = 3
      xs = (\x -> x / n) <$> [(- b * n :: Double) .. b * n]
      ys = (\x -> x / n) <$> [(- b * n :: Double) .. b * n]
      grid = makeAxis xs ys
  gds <- mapM computeGd grid
  let xs' = map (\(x, y, gdr) -> x) gds
      ys' = map (\(x, y, gdr) -> y) gds
      gdrs' = map (\(x, y, gdr) -> gdr) gds
      xDataValue = Numbers xs'
      yDataValue = Numbers ys'
      gdrDataValue = Numbers gdrs'
      xName = pack "x"
      yName = pack "y"
      gdrName = pack "gdr"
      figw = 800
      figh = 800
      dat =
        dataFromColumns [Parse [(xName, FoNumber), (yName, FoNumber), (gdrName, FoNumber)]]
          . dataColumn xName xDataValue
          . dataColumn yName yDataValue
          . dataColumn gdrName gdrDataValue
      enc =
        encoding
          . position X [PName xName, PmType Quantitative]
          . position Y [PName yName, PmType Quantitative]
          . color [MName gdrName, MmType Quantitative]
      vegaPlot = toVegaLite [mark Square [], dat [], enc [], width figw, height figh]
  toHtmlFile "gd-field.html" vegaPlot
