module Main where

import Torch.Data.ActionTransitionSystem (Config (..), testProgram)

main :: IO ()
main =
  let config =
        Config
          { trainingLen = 1000,
            evaluationLen = 100,
            probMaskInput = 0.15,
            probMaskTarget = 0.15,
            maxLearningRate = 0.0005,
            finalLearningRate = 1e-6,
            numEpochs = 100,
            numWarmupEpochs = 10,
            numCooldownEpochs = 10,
            ptFile = "model.pt",
            plotFile = "plot.html"
          }
   in testProgram config
