module Main where

import Torch.Data.ActionTransitionSystem (Config (..), testProgram)
import Torch.Data.Pipeline (MapStyleOptions (..), Sample (Sequential))

main :: IO ()
main =
  let config =
        Config
          { trainingLen = 8192,
            evaluationLen = 96,
            -- trainingLen = 65536,
            -- evaluationLen = 4096,
            probMaskInput = 0.15,
            probMaskTarget = 0.15,
            maxLearningRate = 0.0005,
            finalLearningRate = 1e-6,
            numEpochs = 1000,
            numWarmupEpochs = 10,
            numCooldownEpochs = 10,
            modelCheckpointFile = "modelCheckpoint",
            optimCheckpointFile = "optimCheckpoint",
            plotFile = "plot.html",
            options =
              MapStyleOptions
                { bufferSize = 256,
                  numWorkers = 8,
                  shuffle = Sequential
                }
          }
   in testProgram config
