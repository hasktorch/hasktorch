module Main where

import Test.Hspec (hspec)
import qualified DimnameSpec
import qualified FactorySpec
import qualified FunctionalSpec
import qualified GradSpec
import qualified IndexSpec
import qualified InitializerSpec
import qualified LensSpec
import qualified NNSpec
import qualified OptimSpec
import qualified PipelineSpec
import qualified RandomSpec
import qualified ScriptSpec
import qualified SerializeSpec
import qualified SparseSpec
import qualified TensorSpec
import qualified VisionSpec
import qualified Torch.Distributions.BernoulliSpec
import qualified Torch.Distributions.CategoricalSpec
import qualified Torch.Distributions.ConstraintsSpec
import qualified Torch.Typed.AutogradSpec
import qualified Torch.Typed.AuxiliarySpec
import qualified Torch.Typed.FactoriesSpec
import qualified Torch.Typed.FunctionalSpec0
import qualified Torch.Typed.FunctionalSpec1
import qualified Torch.Typed.FunctionalSpec2
import qualified Torch.Typed.NN.Recurrent.Cell.GRUSpec
import qualified Torch.Typed.NN.Recurrent.Cell.LSTMSpec
import qualified Torch.Typed.NN.Recurrent.GRUSpec
import qualified Torch.Typed.NN.Recurrent.LSTMSpec
import qualified Torch.Typed.NN.TransformerSpec
import qualified Torch.Typed.NNSpec
import qualified Torch.Typed.NamedTensorSpec
import qualified Torch.Typed.OptimSpec
import qualified Torch.Typed.TensorSpec0
import qualified Torch.Typed.TensorSpec1
import qualified Torch.Typed.VisionSpec

main :: IO ()
main = hspec $ do
  DimnameSpec.spec
  FactorySpec.spec
  FunctionalSpec.spec
  GradSpec.spec
  IndexSpec.spec
  InitializerSpec.spec
  LensSpec.spec
  NNSpec.spec
  OptimSpec.spec
  PipelineSpec.spec
  RandomSpec.spec
  ScriptSpec.spec
  SerializeSpec.spec
  SparseSpec.spec
  TensorSpec.spec
  VisionSpec.spec
  Torch.Distributions.BernoulliSpec.spec
  Torch.Distributions.CategoricalSpec.spec
  Torch.Distributions.ConstraintsSpec.spec
  Torch.Typed.AutogradSpec.spec
  Torch.Typed.AuxiliarySpec.spec
  Torch.Typed.FactoriesSpec.spec
  Torch.Typed.FunctionalSpec0.spec
  Torch.Typed.FunctionalSpec1.spec
  Torch.Typed.FunctionalSpec2.spec
  Torch.Typed.NN.Recurrent.Cell.GRUSpec.spec
  Torch.Typed.NN.Recurrent.Cell.LSTMSpec.spec
  Torch.Typed.NN.Recurrent.GRUSpec.spec
  Torch.Typed.NN.Recurrent.LSTMSpec.spec
  Torch.Typed.NN.TransformerSpec.spec
  Torch.Typed.NNSpec.spec
  Torch.Typed.NamedTensorSpec.spec
  Torch.Typed.OptimSpec.spec
  Torch.Typed.TensorSpec0.spec
  Torch.Typed.TensorSpec1.spec
  Torch.Typed.VisionSpec.spec

