{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE CPP #-}
module DataLoading where

#ifdef CUDA
import Torch.Cuda.Double
#else
import Torch.Double
#endif

import Control.Monad
import Control.Arrow (second)
import Control.Exception.Safe (throwString)
import Control.Monad.Trans.Except (runExceptT)
import Data.DList (DList)
import Data.Vector (Vector)
import GHC.TypeLits (KnownNat)
import Torch.Data.Loaders.Cifar10 (Category, rgb2torch)
import Torch.Data.Loaders.RGBVector (Normalize(NegOneToOne))
import Torch.Data.OneHot (onehotf)
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NE
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified Torch.Double.Dynamic as Dynamic



dynTransform :: FilePath -> IO Dynamic
dynTransform f =
  runExceptT (rgb2torch NegOneToOne f) >>= \case
    Left s -> throwString s
    Right (t::Tensor '[3,32,32]) -> do
      pure (asDynamic t)


transform :: FilePath -> IO (Tensor '[3, 32, 32])
transform f =
  runExceptT (rgb2torch NegOneToOne f) >>= \case
    Left s -> throwString s
    Right t -> do
      pure t

-- | potentially lazily loaded data point
type LDatum = (Category, Either FilePath (Tensor '[3, 32, 32]))

-- | potentially lazily loaded dataset
type LDataSet = Vector LDatum

-- | prep cifar10set
prepdata :: Vector (Category, FilePath) -> LDataSet
prepdata = fmap (second Left)

-- | get something usable from a lazy datapoint
getdata :: LDatum -> IO (Category, Tensor '[3, 32, 32])
getdata (c, Right t) = pure (c, t)
getdata (c, Left fp) = (c,) <$> transform fp

-- | force a file into a tensor
forcetensor :: LDatum -> IO LDatum
forcetensor = \case
  (c, Left fp) -> (c,) . Right <$> transform fp
  tens -> pure tens

dataloader
  :: forall batch
  .  All KnownDim '[batch, batch * 10]
  => All KnownNat '[batch, batch * 10]
  => Dim batch
  -> LDataSet
  -> IO (Maybe (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
dataloader d lds
  | V.length lds /= fromIntegral (dimVal d) = pure Nothing
  | otherwise = do
    foo <- V.toList <$> V.mapM getdata lds
    ys <- toYs foo
    xs <- toXs foo
    pure $ Just (ys, xs)
  where
    toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[batch, 10])
    toYs ys =
      unsafeMatrix . fmap (onehotf . fst) $ ys

    toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[batch, 3, 32, 32])
    toXs xs =
      case catArray0 $ fmap (unsqueeze1d (dim :: Dim 0) . snd) (NE.fromList xs) of
        Left s -> throwString s
        Right t -> pure t

dataloader'
  :: forall batch
  .  All KnownDim '[batch, batch * 10]
  => All KnownNat '[batch, batch * 10]
  => Dim batch
  -> Vector (Vector (Category, FilePath))
  -> IO (Vector (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
dataloader' d = mapM go

 where
  go :: Vector (Category, FilePath) -> IO (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32])
  go lzy = do
    ys <- toYs (V.toList $ fst <$> lzy)
    dyns <- mapM dynTransform (NE.fromList . V.toList $ snd <$> lzy)
    forM_ dyns $ \t -> Dynamic._unsqueeze1d t t 0
    case asStatic <$> Dynamic.catArray dyns 0 of
      Left s -> throwString s
      Right res -> pure (ys, res)
   where
    toYs :: [Category] -> IO (Tensor '[batch, 10])
    toYs ys = unsafeMatrix . fmap onehotf $ ys

mkBatches :: Int -> (Vector x) -> [(Vector x)]
mkBatches sz ds = DL.toList $ go mempty ds
 where
  go :: DList (Vector x) -> (Vector x) -> DList (Vector x)
  go bs src =
    if V.null src
    then bs
    else
      let (b, nxt) = V.splitAt sz src
      in go (bs `DL.snoc` b) nxt


mkVBatches :: Int -> (Vector x) -> Vector (Vector x)
mkVBatches a b = V.fromList $ mkBatches a b

