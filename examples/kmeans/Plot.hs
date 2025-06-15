{-# LANGUAGE OverloadedStrings #-}
module Plot where 
import Graphics.Vega.VegaLite
import Data.Text (Text, pack)

plot :: [Double] -> [Double] -> [String] -> IO ()
plot xs ys cs = do 
  let 
    xDataValue = Numbers xs
    yDataValue = Numbers ys
    cDataValue = Strings (map pack cs)
    xName = pack "x"
    yName = pack "y"
    cName = pack "cluster"
    figw = 800
    figh = 800
    dat =
      dataFromColumns []
        . dataColumn xName xDataValue
        . dataColumn yName yDataValue
        . dataColumn cName cDataValue
    enc =
      encoding
        . position X [PName xName, PmType Quantitative]
        . position Y [PName yName, PmType Quantitative]
        . color [MName cName, MmType Nominal]

    bkg = background "rgba(0, 0, 0, 0.05)"
    vegaPlot = toVegaLite [bkg, mark Circle [MTooltip TTEncoding], dat [], enc [], width figw, height figh]
  toHtmlFile "kmeans.html" vegaPlot