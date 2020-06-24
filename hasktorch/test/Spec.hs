import Test.Hspec

import qualified Torch.Typed.NN.Recurrent.Cell.GRUSpec
import qualified Torch.Typed.NN.Recurrent.Cell.LSTMSpec
import qualified Torch.Typed.NN.Recurrent.GRUSpec
import qualified Torch.Typed.NN.Recurrent.LSTMSpec
import qualified Torch.Typed.NN.TransformerSpec
import qualified Torch.Typed.AutogradSpec
import qualified Torch.Typed.AuxSpec
import qualified Torch.Typed.FactoriesSpec
import qualified Torch.Typed.FunctionalSpec
import qualified Torch.Typed.NNSpec
import qualified Torch.Typed.OptimSpec
import qualified Torch.Typed.TensorSpec
import qualified Torch.Typed.VisionSpec
import qualified DimnameSpec
import qualified FactorySpec
import qualified FunctionalSpec
import qualified GradSpec
import qualified InitializerSpec
import qualified NNSpec
import qualified OptimSpec
import qualified RandomSpec
import qualified ScriptSpec
import qualified SerializeSpec
import qualified SparseSpec
import qualified TensorSpec
import qualified VisionSpec

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Torch.Typed.NN.Recurrent.Cell.GRU"  Torch.Typed.NN.Recurrent.Cell.GRUSpec.spec
  describe "Torch.Typed.NN.Recurrent.Cell.LSTM" Torch.Typed.NN.Recurrent.Cell.LSTMSpec.spec
  describe "Torch.Typed.NN.Recurrent.GRU"       Torch.Typed.NN.Recurrent.GRUSpec.spec
  describe "Torch.Typed.NN.Recurrent.LSTM"      Torch.Typed.NN.Recurrent.LSTMSpec.spec
  describe "Torch.Typed.NN.Transformer"         Torch.Typed.NN.TransformerSpec.spec
  describe "Torch.Typed.Autograd"               Torch.Typed.AutogradSpec.spec
  describe "Torch.Typed.Aux"                    Torch.Typed.AuxSpec.spec
  describe "Torch.Typed.Factories"              Torch.Typed.FactoriesSpec.spec
  describe "Torch.Typed.Functional"             Torch.Typed.FunctionalSpec.spec
  describe "Torch.Typed.NN"                     Torch.Typed.NNSpec.spec
  describe "Torch.Typed.Optim"                  Torch.Typed.OptimSpec.spec
  describe "Torch.Typed.Tensor"                 Torch.Typed.TensorSpec.spec
  describe "Torch.Typed.Vision"                 Torch.Typed.VisionSpec.spec
  describe "Dimname"                            DimnameSpec.spec
  describe "Factory"                            FactorySpec.spec
  describe "Functional"                         FunctionalSpec.spec
  describe "Grad"                               GradSpec.spec
  describe "Initializer"                        InitializerSpec.spec
  describe "NN"                                 NNSpec.spec
  describe "Optim"                              OptimSpec.spec
  describe "Random"                             RandomSpec.spec
  describe "Script"                             ScriptSpec.spec
  describe "Serialize"                          SerializeSpec.spec
  describe "Sparse"                             SparseSpec.spec
  describe "Tensor"                             TensorSpec.spec
  describe "Vision"                             VisionSpec.spec
