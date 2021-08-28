{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.NN.Training where

import Control.Monad.IO.Class (MonadIO (..))
import qualified Pipes as P
import qualified Pipes.Prelude as P
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.Optim (Optimizer, stepWithGenerator)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.Random (Generator, SGetGeneratorDevice)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SShape (..), Shape (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar)
import Torch.GraduallyTyped.Tensor.Type (SGetGradient, SGetShape, Tensor, sCheckedGradient, sCheckedShape, withoutGradient)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Train the model for one epoch.
train ::
  forall m model input generatorDevice lossGradient lossLayout lossDataType lossDevice lossShape generatorOutputDevice.
  ( MonadIO m,
    HasStateDict model,
    HasForward model input generatorDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice,
    HasForward model input generatorOutputDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice,
    SGetGeneratorDevice generatorDevice,
    SGetGeneratorDevice generatorOutputDevice,
    SGetGradient lossGradient,
    SGetShape lossShape,
    Catch (lossShape <+> 'Shape '[]),
    Catch (lossGradient <+> 'Gradient 'WithGradient)
  ) =>
  -- | optimizer for the model
  Optimizer model ->
  -- | model specification
  ModelSpec model ->
  -- | stream of training examples
  P.ListT m input ->
  -- | random generator
  Generator generatorDevice ->
  -- | returned is either the original generator or the average training loss and a new generator
  m
    ( Either
        (Generator generatorDevice)
        (Tensor ('Gradient 'WithoutGradient) lossLayout lossDataType lossDevice ('Shape '[]), Generator generatorOutputDevice)
    )
train optim modelSpec examples g = do
  let producer = P.zip (P.enumerate examples) (P.each [0 :: Int ..])
  x <- P.next producer
  case x of
    Left _ -> pure . Left $ g
    Right ((input, iter), producer') -> do
      let step ((loss, _), g') (input', iter') = liftIO $ do
            let forward' model g'' = do
                  (loss', g''') <- forward model input' g''
                  loss'' <- sCheckedShape (SShape SNil) =<< sCheckedGradient (SGradient SWithGradient) loss'
                  pure (loss'', g''')
            (loss', g'') <- stepWithGenerator optim modelSpec forward' g'
            loss'' <- withoutGradient loss'
            pure ((loss + loss'', iter'), g'')
          init' = liftIO $ do
            let forward' model g' = do
                  (loss, g'') <- forward model input g'
                  loss' <- sCheckedShape (SShape SNil) =<< sCheckedGradient (SGradient SWithGradient) loss
                  pure (loss', g'')
            (loss, g') <- stepWithGenerator optim modelSpec forward' g
            loss' <- withoutGradient loss
            pure ((loss', iter), g')
          done ((loss, iter'), g'') = pure . Right $ (loss `divScalar` iter', g'')
      P.foldM step init' done producer'

-- | Evaluate the model on the given examples.
eval ::
  ( MonadIO m,
    HasStateDict model,
    HasForward model input generatorDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice,
    HasForward model input generatorOutputDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice,
    SGetGradient lossGradient,
    SGetShape lossShape,
    Catch (lossShape <+> 'Shape '[]),
    Catch (lossGradient <+> 'Gradient 'WithoutGradient)
  ) =>
  -- | model
  model ->
  -- | stream of examples
  P.ListT m input ->
  -- | random generator
  Generator generatorDevice ->
  -- | returned is either the original generator or the average evaluation loss and a new generator
  m
    ( Either
        (Generator generatorDevice)
        (Tensor ('Gradient 'WithoutGradient) lossLayout lossDataType lossDevice ('Shape '[]), Generator generatorOutputDevice)
    )
eval model examples g = do
  let producer = P.zip (P.enumerate examples) (P.each [0 :: Int ..])
  x <- P.next producer
  case x of
    Left _ -> pure . Left $ g
    Right ((input, iter), producer') -> do
      let step ((loss, _), g') (input', iter') = liftIO $ do
            (loss', g'') <- forward model input' g'
            loss'' <- sCheckedShape (SShape SNil) =<< sCheckedGradient (SGradient SWithoutGradient) loss'
            pure ((loss + loss'', iter'), g'')
          init' = liftIO $ do
            (loss, g') <- forward model input g
            loss' <- sCheckedShape (SShape SNil) =<< sCheckedGradient (SGradient SWithoutGradient) loss
            pure ((loss', iter), g')
          done ((loss, iter'), g'') = pure . Right $ (loss `divScalar` iter', g'')
      P.foldM step init' done producer'
