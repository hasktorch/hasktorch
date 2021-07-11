{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Plot where

import Graphics.Vega.VegaLite hiding (sample, shape)
import Torch

scatter x y = do
  let x' :: [Double] = asValue . toDType Double $ x
      y' :: [Double] = asValue . toDType Double $ y
      dat =
        dataFromColumns [Parse [("x", FoNumber), ("y", FoNumber)]]
          . dataColumn "x" (Numbers x')
          . dataColumn "y" (Numbers y')
      enc =
        encoding
          . position X [PName ("x"), PmType Quantitative]
          . position Y [PName ("y"), PmType Quantitative]
  pure $ toVegaLite [mark Circle [MTooltip TTEncoding], dat [], enc [], width 400, height 400]

histogram x = do
  let x' :: [Double] = asValue . toDType Double $ x
      dat = dataFromColumns [Parse [("x", FoNumber)]] . dataColumn "x" (Numbers x')
      enc =
        ( encoding
            . position X [PName "x", PmType Quantitative, PBin [Step 0.005]]
            . position Y [PAggregate Count, PmType Quantitative]
            . tooltip [TName "x"]
        )
  pure $ toVegaLite [mark Bar [], dat [], enc []]

strip x = do
  let x' :: [Double] = asValue . toDType Double $ x
      dat = dataFromColumns [Parse [("x", FoNumber)]] . dataColumn "x" (Numbers x')
      enc =
        encoding
          . position X [PName "x", PmType Quantitative]
          . tooltip [TName "x"]
  pure $ toVegaLite [dat [], mark Tick [MOpacity 0.1], enc []]
