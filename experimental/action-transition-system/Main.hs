module Main where

import Torch.Data.ActionTransitionSystem (Config (..), testProgram)
import Torch.Data.Pipeline (mapStyleOpts)

main :: IO ()
main =
  let config =
        Config
          { trainingLen = 65536,
            evaluationLen = 4096,
            probMaskInput = 0.15,
            probMaskTarget = 0.15,
            maxLearningRate = 0.0005,
            finalLearningRate = 1e-6,
            numEpochs = 1000,
            numWarmupEpochs = 10,
            numCooldownEpochs = 10,
            ptFile = "model.pt",
            plotFile = "plot.html",
            options = mapStyleOpts 8
          }
   in testProgram config
