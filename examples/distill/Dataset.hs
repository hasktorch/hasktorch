module Dataset where

import Torch hiding (take)
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

-- This is a placeholder for this example until we have a more formal data loader abstraction
--
class MockDataset d where
  getItem ::
    d ->
    Int -> -- index
    Int -> -- batchSize
    IO (Tensor, Tensor) -- input, label

data MNIST = MNIST
  { dataset :: V.MnistData,
    idxList :: [Int]
  }

instance MockDataset MNIST where
  getItem mnistData index n = do
    let currIndex = index
    let idx = take n (drop (currIndex + n) (idxList mnistData))
    let input = V.getImages' n mnistDataDim (dataset mnistData) idx
    let label = V.getLabels' n (dataset mnistData) idx
    pure (input, label)

mnistDataDim = 784

-- | Load MNIST data as dataset abstraction
loadMNIST dataLocation = do
  (train, test) <- V.initMnist dataLocation
  let mnistTrain =
        MNIST
          { dataset = train,
            idxList = V.randomIndexes (V.length train)
          }
  let mnistTest =
        MNIST
          { dataset = test,
            idxList = V.randomIndexes (V.length train)
          }
  pure (mnistTrain, mnistTest)
