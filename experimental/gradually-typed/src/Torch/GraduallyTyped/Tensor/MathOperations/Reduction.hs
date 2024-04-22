{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Reduction where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.State (execState, modify)
import Data.Bifunctor (Bifunctor (first), second)
import Data.Foldable (for_)
import Data.Kind (Constraint)
import qualified Data.Set as Set
import Data.Singletons (SingI (..), SingKind (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (ErrorMessage, Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked)
import Torch.GraduallyTyped.Shape.Class (ReplaceDimSizeImplF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SSelectDim, SSelectDims, SelectDim (..), SelectDims (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import qualified Torch.Internal.Cast as ATen (cast1, cast3)
import qualified Torch.Internal.Class as ATen (Castable (cast), uncast)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Type as ATen (Tensor)
import Type.Errors.Pretty (type (%), type (<>))
import Prelude hiding (all, any)

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped
-- >>> import Prelude hiding (all, any)

type ReductionErrorMessage :: Symbol -> By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> ErrorMessage

type ReductionErrorMessage reduction by dims =
  "Cannot apply '" <> reduction <> "' on the dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type ReductionCheckF ::
  Symbol ->
  By Symbol Nat ->
  [Dim (Name Symbol) (Size Nat)] ->
  Maybe [Dim (Name Symbol) (Size Nat)] ->
  [Dim (Name Symbol) (Size Nat)]
type family ReductionCheckF reduction by dims result where
  ReductionCheckF reduction by dims 'Nothing = TypeError (ReductionErrorMessage reduction by dims)
  ReductionCheckF _ _ _ ('Just dims') = dims'

type BoolReductionF ::
  Symbol ->
  SelectDim (By Symbol Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]
type family BoolReductionF reduction selectDim shape where
  BoolReductionF _ 'UncheckedSelectDim _ = 'UncheckedShape
  BoolReductionF _ _ 'UncheckedShape = 'UncheckedShape
  BoolReductionF reduction ('SelectDim by) ('Shape dims) = 'Shape (ReductionCheckF reduction by dims (ReplaceDimSizeImplF by dims ('Size 1)))

-- | Tests if all element in input evaluates to True.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> shape = SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @4 :|: SNil
-- >>> (t, _) <- sRandn (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) shape) g
-- >>> t' <- all =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape '[])
all ::
  forall requiresGradient layout device dataType shape m.
  MonadThrow m =>
  Tensor requiresGradient layout device dataType shape ->
  m (Tensor requiresGradient layout device ('DataType 'Bool) ('Shape '[]))
all = unsafeThrowableIO . ATen.cast1 ATen.all_t

-- | Reduces each row of the input tensor in the selected dimension to True if all elements in the row evaluate to True and False otherwise.
-- For a version that accepts non-singleton parameters see 'allDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> shape = SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @4 :|: SNil
-- >>> (t, _) <- sRandn (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) shape) g
-- >>> t' <- sAllDim (SSelectDim (SByIndex @1)) =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1)])
--
-- >>> sAllDim (SUncheckedSelectDim (ByIndex 3)) t
-- *** Exception: HasktorchException "Exception: Dimension out of range (expected to be in range of [-2, 1], but got 3)...
sAllDim ::
  forall selectDim gradient layout device dataType shape shape' m.
  (MonadThrow m, shape' ~ BoolReductionF "all" selectDim shape, Catch shape') =>
  SSelectDim selectDim ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device ('DataType 'Bool) shape')
sAllDim by tensor = unsafeThrowableIO $ case forgetIsChecked $ fromSing by of
  ByName name ->
    ATen.cast3
      ATen.all_tnb
      tensor
      name
      True -- keepDim
  ByIndex index ->
    ATen.cast3
      ATen.all_tlb
      tensor
      (fromInteger index :: Int)
      True -- keepDim

type AllDimF :: SelectDim (By Symbol Nat) -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]

type AllDimF selectDim shape = BoolReductionF "all" selectDim shape

-- | Reduces each row of the input tensor in the selected dimension to True if all elements in the row evaluate to True and False otherwise.
-- For a version that accepts singleton parameters see 'sAllDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> type Shape' = 'Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 4) ]
-- >>> (t, _) <- randn @('Gradient 'WithoutGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @Shape' g
-- >>> t' <- allDim @('SelectDim ('ByIndex 1)) =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1)])
allDim ::
  forall selectDim gradient layout device dataType shape shape' m.
  (SingI selectDim, MonadThrow m, shape' ~ AllDimF selectDim shape, Catch shape') =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device ('DataType 'Bool) shape')
allDim = sAllDim (sing @selectDim)

-- | Tests if any element in input evaluates to True.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> shape = SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @4 :|: SNil
-- >>> (t, _) <- sRandn (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) shape) g
-- >>> t' <- any =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape '[])
any ::
  forall requiresGradient layout device dataType shape m.
  MonadThrow m =>
  Tensor requiresGradient layout device dataType shape ->
  m (Tensor requiresGradient layout device ('DataType 'Bool) ('Shape '[]))
any = unsafeThrowableIO . ATen.cast1 ATen.any_t

type AnyDimF :: SelectDim (By Symbol Nat) -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]

type AnyDimF selectDim shape = BoolReductionF "any" selectDim shape

-- | Reduces each row of the input tensor in the selected dimension to True if any element in the row evaluates to True and False otherwise.
-- For a version that accepts non-singleton parameters see 'anyDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> shape = SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @4 :|: SNil
-- >>> (t, _) <- sRandn (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) shape) g
-- >>> t' <- sAnyDim (SSelectDim (SByIndex @1)) =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1)])
--
-- >>> sAnyDim (SUncheckedSelectDim (ByIndex 3)) t
-- *** Exception: HasktorchException "Exception: Dimension out of range (expected to be in range of [-2, 1], but got 3)...
sAnyDim ::
  forall selectDim gradient layout device shape dataType shape' m.
  (MonadThrow m, shape' ~ AnyDimF selectDim shape, Catch shape') =>
  SSelectDim selectDim ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device ('DataType 'Bool) shape')
sAnyDim selectDim tensor = unsafeThrowableIO $
  case forgetIsChecked $ fromSing selectDim of
    ByName name ->
      ATen.cast3
        ATen.any_tnb
        tensor
        name
        True -- keepDim
    ByIndex index ->
      ATen.cast3
        ATen.any_tlb
        tensor
        (fromInteger index :: Int)
        True -- keepDim

-- | Reduces each row of the input tensor in the selected dimension to True if any element in the row evaluates to True and False otherwise.
-- For a version that accepts singleton parameters see 'sAnyDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> type Shape' = 'Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 4) ]
-- >>> (t, _) <- randn @('Gradient 'WithoutGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @Shape' g
-- >>> t' <- anyDim @('SelectDim ('ByIndex 1)) =<< bool t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Bool)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1)])
anyDim ::
  forall selectDim gradient layout device dataType shape shape' m.
  (SingI selectDim, MonadThrow m, shape' ~ AnyDimF selectDim shape, Catch shape') =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device ('DataType 'Bool) shape')
anyDim = sAnyDim (sing @selectDim)

type family MeanSelectDimsF (bys :: [By Symbol Nat]) (dims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  MeanSelectDimsF '[] dims = dims
  MeanSelectDimsF (by ': bys) dims = MeanSelectDimsF bys (ReductionCheckF "mean" by dims (ReplaceDimSizeImplF by dims ('Size 1)))

type family MeanF (selectDims :: SelectDims [By Symbol Nat]) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  MeanF 'UncheckedSelectDims _ = 'UncheckedShape
  MeanF _ 'UncheckedShape = 'UncheckedShape
  MeanF ('SelectDims bys) ('Shape dims) = 'Shape (MeanSelectDimsF bys dims)

-- | Reduces the mean value over each row of the input tensor in the dimensions selected by 'selectDims'.
-- For a version that accepts non-singleton parameters see 'meanDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> shape = SShape $ SName @"batch" :&: SSize @8 :|: SName @"width" :&: SSize @224 :|: SName @"height" :&: SSize @224 :|: SNil
-- >>> (t, _) <- sRandn (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) shape) g
-- >>> t' <- sMeanDims (SSelectDims $ SByName @"width" :|: SByName @"height" :|: SNil) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 8), 'Dim ('Name "width") ('Size 1),
--            'Dim ('Name "height") ('Size 1)])
--
-- >>> sMeanDims (SUncheckedSelectDims [ByName "feature"]) t
-- *** Exception: HasktorchException "Exception: Name 'feature' not found in Tensor['batch', 'width', 'height']...
sMeanDims ::
  forall selectDims gradient layout device dataType shape shape' m.
  (MonadThrow m, shape' ~ MeanF selectDims shape, Catch shape') =>
  SSelectDims selectDims ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
sMeanDims bys tensor =
  let bys' = forgetIsChecked $ fromSing bys
      (names, indexes) = flip execState (Set.empty, Set.empty) $ do
        for_ bys' $ \by -> do
          case by of
            ByName name -> modify . first $ Set.insert name
            ByIndex index -> modify . second $ Set.insert index
   in unsafeThrowableIO $ do
        case (names, indexes) of
          (names', indexes')
            | Set.null names' && Set.null indexes' ->
              do
                t :: ForeignPtr ATen.Tensor <- ATen.cast tensor pure
                ATen.uncast t pure
            | Set.null names' ->
              ATen.cast1 (meanIndexes indexes') tensor
            | Set.null indexes' ->
              ATen.cast1 (meanNames names') tensor
            | otherwise ->
              do
                t' :: ForeignPtr ATen.Tensor <- ATen.cast1 (meanIndexes indexes') tensor
                ATen.cast1 (meanNames names') t'
  where
    meanNames :: Set.Set String -> ForeignPtr ATen.Tensor -> IO (ForeignPtr ATen.Tensor)
    meanNames names tensor' =
      ATen.cast3
        ATen.mean_tNb
        tensor'
        (Set.toList names)
        True -- keepDim
    meanIndexes :: Set.Set Integer -> ForeignPtr ATen.Tensor -> IO (ForeignPtr ATen.Tensor)
    meanIndexes indexes tensor' =
      ATen.cast3
        ATen.mean_tlb
        tensor'
        (Set.toList indexes)
        True -- keepDim

-- | Reduce the mean value over each row of the input tensor in the dimensions selected by 'selectDims'.
-- For a version that accepts singleton parameters see 'sMeanDim'.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> type Shape' = 'Shape '[ 'Dim ('Name "batch") ('Size 8), 'Dim ('Name "feature") ('Size 4) ]
-- >>> (t, _) <- randn @('Gradient 'WithoutGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @Shape' g
-- >>> t' <- meanDims @('SelectDims '[ 'ByName "feature" ]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 8), 'Dim ('Name "feature") ('Size 1)])
meanDims ::
  forall selectDims gradient layout device dataType shape shape' m.
  (SingI selectDims, MonadThrow m, shape' ~ MeanF selectDims shape, Catch shape') =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
meanDims = sMeanDims (sing @selectDims)

type DimPositiveMessage :: Symbol -> Dim (Name Symbol) (Size Nat) -> ErrorMessage

type DimPositiveMessage reduction dim =
  "Cannot apply '" <> reduction <> "' because the dimension"
    % ""
    % "    '" <> dim <> "'"
    % ""
    % "is not positive."

type DimPositiveF :: Symbol -> Dim (Name Symbol) (Size Nat) -> Constraint
type family DimPositiveF reduction dim where
  DimPositiveF _ ('Dim _ 'UncheckedSize) = ()
  DimPositiveF reduction ('Dim name ('Size 0)) = TypeError (DimPositiveMessage reduction ('Dim name ('Size 0)))
  DimPositiveF _ ('Dim _ ('Size _size)) = ()

type AllDimsPositiveImplF :: Symbol -> [Dim (Name Symbol) (Size Nat)] -> Constraint
type family AllDimsPositiveImplF reduction dims where
  AllDimsPositiveImplF _ '[] = ()
  AllDimsPositiveImplF reduction (dim ': dims) = (DimPositiveF reduction dim, AllDimsPositiveImplF reduction dims)

type AllDimsPositiveF :: Symbol -> Shape [Dim (Name Symbol) (Size Nat)] -> Constraint
type family AllDimsPositiveF reduction shape where
  AllDimsPositiveF _ 'UncheckedShape = ()
  AllDimsPositiveF reduction ('Shape dims) = AllDimsPositiveImplF reduction dims

type MeanAllCheckF :: Shape [Dim (Name Symbol) (Size Nat)] -> Constraint

type MeanAllCheckF shape = AllDimsPositiveF "meanAll" shape

-- | Reduces a tensor by calculating the mean value over all dimensions.
meanAll ::
  forall gradient layout device dataType shape.
  MeanAllCheckF shape =>
  Tensor gradient layout device dataType shape ->
  Tensor gradient layout device dataType ('Shape '[])
meanAll = unsafePerformIO . ATen.cast1 ATen.mean_t

type ArgmaxF :: SelectDim (By Symbol Nat) -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]

type ArgmaxF selectDim shape = BoolReductionF "argmax" selectDim shape

-- | Argmax of a tensor given a dimension.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> spec = TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SNoName :&: SSize @2 :|: SNoName :&: SSize @5 :|: SNil)
-- >>> (t, _) <- sRandn spec g
-- >>> r <- argmax (SSelectDim $ SByIndex @1) t
-- >>> :type r
-- r :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType Int64)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1)])
-- >>> r
-- Tensor Int64 [2,1] [[ 0],
--                     [ 2]]
argmax ::
  forall selectDims gradient layout device dataType shape shape' m.
  (MonadThrow m, shape' ~ ArgmaxF selectDims shape, Catch shape') =>
  SSelectDim selectDims ->
  Tensor gradient layout device dataType shape ->
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int64) shape')
argmax selectDim input =
  unsafeThrowableIO $
    let by = forgetIsChecked . fromSing $ selectDim
     in case by of
          ByName name -> undefined
          ByIndex index ->
            ATen.cast3
              ATen.argmax_tlb
              input
              (fromInteger index :: Int)
              True -- keepDim

type MaxAllCheckF :: Shape [Dim (Name Symbol) (Size Nat)] -> Constraint

type MaxAllCheckF shape = AllDimsPositiveF "maxAll" shape

maxAll ::
  forall gradient layout device dataType shape.
  MaxAllCheckF shape =>
  Tensor gradient layout device dataType shape ->
  Tensor gradient layout device dataType ('Shape '[])
maxAll = unsafePerformIO . ATen.cast1 ATen.max_t
