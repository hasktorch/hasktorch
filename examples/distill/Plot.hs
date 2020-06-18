{-# LANGUAGE OverloadedStrings #-}

module Plot where

import Graphics.Vega.VegaLite hiding (sample, shape)
import Torch

scatter x y = do 
    let x' = asValue . toDType Double $ x
        y' = asValue . toDType Double $ y
    let dat = dataFromColumns [Parse [("x", FoNumber), ("y", FoNumber)]]
          . dataColumn "x" (Numbers x')
          . dataColumn "y" (Numbers y')
    let enc = encoding
          . position X [PName ("x"), PmType Quantitative]
          . position Y [PName ("y"), PmType Quantitative]
    let vegaPlot = toVegaLite [mark Circle [MTooltip TTEncoding], dat [], enc [], width 400, height 400]
    toHtmlFile "plot.html" vegaPlot
