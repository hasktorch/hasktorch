{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Reduction where

import Control.Monad.State (execState, modify)
import Data.Bifunctor (Bifunctor (first), second)
import Data.Foldable (for_)
import qualified Data.Set as Set
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Shape.Class (ReplaceDimSizeImplF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SelectDims (..), Shape (..), Size (..), WithSelectDimsC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast1, cast3)
import Torch.Internal.Class (Castable (cast), uncast)
import qualified Torch.Internal.Managed.Native as ATen (mean_tNb, mean_tlb)
import qualified Torch.Internal.Type as ATen (Tensor)
import Type.Errors.Pretty (type (%), type (<>))

type MeanErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot apply mean on the dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family MeanCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  MeanCheckF by dims 'Nothing = TypeError (MeanErrorMessage by dims)
  MeanCheckF _ _ ( 'Just dims') = dims'

type family MeanSelectDimsF (bys :: [By Symbol Nat]) (dims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  MeanSelectDimsF '[] dims = dims
  MeanSelectDimsF (by ': bys) dims = MeanSelectDimsF bys (MeanCheckF by dims (ReplaceDimSizeImplF by dims ( 'Size 1)))

type family MeanF (selectDims :: SelectDims [By Symbol Nat]) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  MeanF 'UncheckedSelectDims _ = 'UncheckedShape
  MeanF _ 'UncheckedShape = 'UncheckedShape
  MeanF ( 'SelectDims bys) ( 'Shape dims) = 'Shape (MeanSelectDimsF bys dims)

mean ::
  forall selectDims requiresGradient layout device dataType shape.
  WithSelectDimsC
    selectDims
    ( Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (MeanF selectDims shape)
    ) =>
  WithSelectDimsF
    selectDims
    ( Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (MeanF selectDims shape)
    )
mean = withSelectDims @selectDims
  @( Tensor requiresGradient layout device dataType shape ->
     Tensor requiresGradient layout device dataType (MeanF selectDims shape)
   )
  $ \bys tensor ->
    let (names, indexes) = flip execState (Set.empty, Set.empty) $ do
          for_ bys $ \by -> do
            case by of
              ByName name -> modify . first $ Set.insert name
              ByIndex index -> modify . second $ Set.insert index
     in unsafePerformIO $ do
          case (names, indexes) of
            (names, indexes)
              | Set.null names && Set.null indexes ->
                do
                  t :: ForeignPtr ATen.Tensor <- cast tensor pure
                  uncast t pure
              | Set.null names ->
                cast1 (meanIndexes indexes) tensor
              | Set.null indexes ->
                cast1 (meanNames names) tensor
              | otherwise ->
                do
                  t' :: ForeignPtr ATen.Tensor <- cast1 (meanIndexes indexes) tensor
                  cast1 (meanNames names) t'
  where
    meanNames :: Set.Set String -> ForeignPtr ATen.Tensor -> IO (ForeignPtr ATen.Tensor)
    meanNames names tensor =
      cast3
        ATen.mean_tNb
        tensor
        (Set.toList names)
        True -- keepDim
    meanIndexes :: Set.Set Integer -> ForeignPtr ATen.Tensor -> IO (ForeignPtr ATen.Tensor)
    meanIndexes indexes tensor =
      cast3
        ATen.mean_tlb
        tensor
        (Set.toList indexes)
        True -- keepDim
