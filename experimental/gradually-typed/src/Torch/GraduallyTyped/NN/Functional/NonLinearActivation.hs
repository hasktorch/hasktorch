{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Functional.NonLinearActivation where

import Control.Monad.Catch (MonadThrow)
import Data.Singletons (SingKind (..))
import GHC.TypeLits (Nat, Symbol, TypeError)
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked)
import Torch.GraduallyTyped.Shape (By (..), Dim (..), GetDimImplF, Name (..), SSelectDim (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast2)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

type SoftMaxErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply softmax on the dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family SoftmaxCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe (Dim (Name Symbol) (Size Nat))) :: [Dim (Name Symbol) (Size Nat)] where
  SoftmaxCheckF by dims 'Nothing = TypeError (SoftMaxErrorMessage by dims)
  SoftmaxCheckF _ dims ('Just _) = dims

type family SoftmaxF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  SoftmaxF 'UncheckedSelectDim _ = 'UncheckedShape
  SoftmaxF _ 'UncheckedShape = 'UncheckedShape
  SoftmaxF ('SelectDim by) ('Shape dims) = 'Shape (SoftmaxCheckF by dims (GetDimImplF by dims))

-- | Applies the softmax function that is defined as:
--
-- \[
-- \mathrm{Softmax}(\mathrm{input}_{i}) = \frac{\exp\left(\mathrm{input}_{i}\right)}{\sum_j \exp\left(\mathrm{input}_{j}\right)}
-- \]
--
-- Softmax is applied to all slices along 'selectDim',
-- and will re-scale them so that the elements lie in the range \([0, 1]\) and sum to \(1\):
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> (input, _) <- sRandn (TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)) g
-- >>> result <- softmax (SSelectDim (SByName @"feature")) input
-- >>> :type result
-- result
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
softmax,
  logSoftmax ::
    forall selectDim gradient layout device dataType shape shape' m.
    (MonadThrow m, shape' ~ SoftmaxF selectDim shape, Catch shape') =>
    SSelectDim selectDim ->
    Tensor gradient layout device dataType shape ->
    m (Tensor gradient layout device dataType shape')
softmax selectDim tensor =
  case forgetIsChecked (fromSing selectDim) of
    ByName name -> unsafeThrowableIO $ cast2 ATen.softmax_tn tensor name
    ByIndex index -> unsafeThrowableIO $ cast2 ATen.softmax_tl tensor (fromInteger index :: Int)
logSoftmax selectDim tensor =
  case forgetIsChecked (fromSing selectDim) of
    ByName name -> unsafeThrowableIO $ cast2 ATen.log_softmax_tn tensor name
    ByIndex index -> unsafeThrowableIO $ cast2 ATen.log_softmax_tl tensor (fromInteger index :: Int)
