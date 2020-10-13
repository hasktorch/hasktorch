{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.IndexingSlicingJoining where

import Data.Kind (Type)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType, UnifyDataTypeF)
import Torch.GraduallyTyped.Device (Device, DeviceType, UnifyDeviceF)
import Torch.GraduallyTyped.Layout (Layout, LayoutType, UnifyLayoutF)
import Torch.GraduallyTyped.Prelude (MapMaybe)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient, UnifyRequiresGradientF)
import Torch.GraduallyTyped.Shape (AddDimF, Dim (..), DimBy (..), GetDimByF, IsAnyDimBy, ReplaceDimByF, ReplaceDimByImplF, Shape (..), UnifyShapeF, WithDimByC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorF)
import Torch.HList (HList)
import Torch.Internal.Cast (cast2)
import Torch.Internal.Class (Castable)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Type as ATen
import Type.Errors.Pretty (ToErrorMessage, type (%), type (<>))

class HasCat (dimBy :: DimBy Symbol Nat) k (c :: k -> Type) (a :: k) where
  type CatF dimBy a c :: Type

  -- | Concatenates the given sequence of seq tensors in the given dimension.
  -- All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  --
  -- >>> t <- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSizedDim "batch" 32, 'NamedSizedDim "feature" 8])
  -- >>> :type cat @('DimByName "feature") [t]
  -- cat @('DimByName "feature") [t]
  -- :: Tensor
  --      'Dependent
  --      ('Layout 'Dense)
  --      ('Device ('CUDA 0))
  --      ('DataType 'Half)
  --      ('Shape '[ 'NamedSizedDim "batch" 32, 'AnyDim])
  -- >>> :type cat @('DimByIndex 0) [t]
  -- cat @('DimByIndex 0) [t]
  --   :: Tensor
  --        'Dependent
  --        ('Layout 'Dense)
  --        ('Device ('CUDA 0))
  --        ('DataType 'Half)
  --        ('Shape '[ 'AnyDim, 'NamedSizedDim "feature" 8])
  -- >>> :type cat @'AnyDimBy (DimByIndex 0) [t]
  -- cat @'AnyDimBy (DimByIndex 0) [t]
  --   :: Tensor
  --        'Dependent
  --        ('Layout 'Dense)
  --        ('Device ('CUDA 0))
  --        ('DataType 'Half)
  --        'AnyShape
  cat :: WithDimByF (IsAnyDimBy dimBy) (c a -> CatF dimBy a c)

type family CatListImplF (dimBy :: DimBy Symbol Nat) (tensor :: Type) :: Maybe Type where
  CatListImplF 'AnyDimBy (Tensor requiresGradient layout device dataType _) = 'Just (Tensor requiresGradient layout device dataType 'AnyShape)
  CatListImplF dimBy (Tensor requiresGradient layout device dataType shape) = MapMaybe (Tensor requiresGradient layout device dataType) (MapMaybe 'Shape (ReplaceDimByImplF dimBy shape 'AnyDim))

type CheckSpellingMessage = "Check the spelling of named dimensions, and make sure the number of dimensions is correct."

type family CatListCheckF (dimBy :: DimBy Symbol Nat) (tensor :: Type) (result :: Maybe Type) :: Type where
  CatListCheckF dimBy (Tensor _ _ _ _ shape) 'Nothing =
    TypeError
      ( "Cannot concatenate the dimension"
          % ""
          % "    " <> dimBy
          % ""
          % "for tensors of shape"
          % ""
          % "    " <> shape <> "."
          % ""
          % CheckSpellingMessage
      )
  CatListCheckF _ _ ( 'Just result) = result

type CatListF dimBy tensor = CatListCheckF dimBy tensor (CatListImplF dimBy tensor)

instance
  ( WithDimByC (IsAnyDimBy dimBy) dimBy ([Tensor requiresGradient layout device dataType shape] -> CatListF dimBy (Tensor requiresGradient layout device dataType shape)),
    Castable (CatListF dimBy (Tensor requiresGradient layout device dataType shape)) (ForeignPtr ATen.Tensor)
  ) =>
  HasCat dimBy Type [] (Tensor requiresGradient layout device dataType shape)
  where
  type CatF dimBy (Tensor requiresGradient layout device dataType shape) [] = CatListF dimBy (Tensor requiresGradient layout device dataType shape)
  cat =
    withDimBy @(IsAnyDimBy dimBy) @dimBy @([Tensor requiresGradient layout device dataType shape] -> CatF dimBy (Tensor requiresGradient layout device dataType shape) []) $
      \dimBy tensors -> case dimBy of
        AnyDimBy -> error "A concatenation dimension must be specified, but none was given."
        DimByName name -> undefined
        DimByIndex index -> unsafePerformIO $ cast2 ATen.cat_ll tensors (fromInteger index :: Int)

type family
  CatHListImplF
    (dimBy :: DimBy Symbol Nat)
    (tensors :: [Type])
    (acc :: Maybe (RequiresGradient, Layout LayoutType, Device (DeviceType Nat), DataType DType, Shape [Dim Symbol Nat])) ::
    Type
  where
  CatHListImplF _ '[] 'Nothing = TypeError (ToErrorMessage "Cannot concatenate an empty list of tensors.")
  CatHListImplF _ '[] ( 'Just '(requiresGradient, layout, device, dataType, shape)) = TensorF '(requiresGradient, layout, device, dataType, shape)
  CatHListImplF dimBy (Tensor requiresGradient layout device dataType shape ': tensors) 'Nothing =
    CatHListImplF dimBy tensors ( 'Just '(requiresGradient, layout, device, dataType, shape))
  CatHListImplF dimBy (Tensor requiresGradient layout device dataType shape ': tensors) ( 'Just '(requiresGradient', layout', device', dataType', shape')) =
    CatHListImplF
      dimBy
      tensors
      ( 'Just
          '( UnifyRequiresGradientF requiresGradient requiresGradient',
             UnifyLayoutF layout layout',
             UnifyDeviceF device device',
             UnifyDataTypeF dataType dataType',
             ReplaceDimByF
               dimBy
               (UnifyShapeF (ReplaceDimByF dimBy shape 'AnyDim) (ReplaceDimByF dimBy shape' 'AnyDim))
               (AddDimF (GetDimByF dimBy shape) (GetDimByF dimBy shape))
           )
      )
  CatHListImplF _ (x ': _) _ =
    TypeError
      ( "Cannot concatenate because"
          % ""
          % "    '" <> x <> "'"
          % ""
          % "is not a tensor type."
      )

type CatHListF dimBy tensors = CatHListImplF dimBy tensors 'Nothing

instance
  ( WithDimByC (IsAnyDimBy dimBy) dimBy (HList tensors -> CatHListF dimBy tensors),
    Castable (CatHListF dimBy tensors) (ForeignPtr ATen.Tensor),
    Castable (HList tensors) (ForeignPtr ATen.TensorList)
  ) =>
  HasCat dimBy [Type] HList tensors
  where
  type CatF dimBy tensors HList = CatHListF dimBy tensors
  cat =
    withDimBy @(IsAnyDimBy dimBy) @dimBy @(HList tensors -> CatF dimBy tensors HList) $
      \dimBy tensors -> case dimBy of
        AnyDimBy -> error "A concatenation dimension must be specified, but none was given."
        DimByName name -> undefined
        DimByIndex index -> unsafePerformIO $ cast2 ATen.cat_ll tensors (fromInteger index :: Int)

-- data ListOf a = Homogeneous a | Heterogeneous a

-- class ListOfC (listOf :: ListOf a) (c :: a -> Type) (f :: Type) where
--   type ListOfF listOf f :: Type
--   withListOf :: (c a -> f) -> ListOfF isHomogeneous f

-- instance ListOfC ('Homogeneous a) [] f where
--   type ListOfF ('Homogeneous a) f = [a] -> f
--   withListOf = id

-- instance ListOfC ('Heterogeneous a) HList f where
--   type ListOfF ('Heterogeneous a) f = [a] -> f
--   withListOf f = f (dimByVal @dimBy)

-- -- | Concatenates the given sequence of seq tensors in the given dimension.
-- -- All tensors must either have the same shape (except in the concatenating dimension) or be empty.
-- cat
--   :: Dim -- ^ dimension
--   -> [Tensor] -- ^ list of tensors to concatenate
--   -> Tensor -- ^ output tensor
-- cat (Dim d) tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors d

-- cat ::
--   forall dim shape dtype device tensors.
--   ( KnownNat dim,
--     '(shape, dtype, device) ~ Cat dim tensors,
--     ATen.Castable (HList tensors) [D.ATenTensor]
--   ) =>
--   -- | input list of tensors
--   HList tensors ->
--   -- | output tensor
--   Tensor device dtype shape
-- cat tensors = unsafePerformIO $ ATen.cast2 ATen.Managed.cat_ll tensors (natValI @dim :: Int)
