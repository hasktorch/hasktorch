{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Unify where

import Data.Type.Bool (type (&&))
import GHC.TypeLits (Symbol, TypeError)
import GHC.TypeNats (Nat)
import Torch.GraduallyTyped.DType (DType, DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Type.Errors.Pretty (type (%), type (<>))

type (<+>) :: forall k. k -> k -> k

-- | @a <+> b@ unifies @a@ and @b@.
-- Think of it as a kind-level monoid.
type family (<+>) a b where
  (<+>) (a :: k) (b :: k) = Unify k a b

infixr 8 <+>

-- | Desugared kind unification.
--
-- TODO: add data type unification of scalar (Haskell) data types and those of kind @DataType@.
-- Perhaps convert the scalar (Haskell) data type first to a @DataType@ so that the kinds are aligned.
type Unify :: forall k -> k -> k -> k
type family Unify k a b where
  Unify _ a a = a
  Unify (Gradient RequiresGradient) 'UncheckedGradient _ = 'UncheckedGradient
  Unify (Gradient RequiresGradient) _ 'UncheckedGradient = 'UncheckedGradient
  Unify (Gradient RequiresGradient) ('Gradient requiresGradient) ('Gradient requiresGradient') = TypeError (UnifyRequiresGradientMessage requiresGradient requiresGradient')
  Unify (Layout LayoutType) 'UncheckedLayout _ = 'UncheckedLayout
  Unify (Layout LayoutType) _ 'UncheckedLayout = 'UncheckedLayout
  Unify (Layout LayoutType) ('Layout layoutType) ('Layout layoutType') = TypeError (UnifyLayoutErrorMessage layoutType layoutType')
  Unify (Device (DeviceType Nat)) 'UncheckedDevice _ = 'UncheckedDevice
  Unify (Device (DeviceType Nat)) _ 'UncheckedDevice = 'UncheckedDevice
  Unify (Device (DeviceType Nat)) ('Device deviceType) ('Device deviceType') = TypeError (UnifyDeviceErrorMessage deviceType deviceType')
  Unify (DataType DType) 'UncheckedDataType _ = 'UncheckedDataType
  Unify (DataType DType) _ 'UncheckedDataType = 'UncheckedDataType
  Unify (DataType DType) ('DataType dType) ('DataType dType') = TypeError (UnifyDataTypeErrorMessage dType dType')
  Unify (Shape [Dim (Name Symbol) (Size Nat)]) 'UncheckedShape _ = 'UncheckedShape
  Unify (Shape [Dim (Name Symbol) (Size Nat)]) _ 'UncheckedShape = 'UncheckedShape
  Unify (Shape [Dim (Name Symbol) (Size Nat)]) ('Shape dims) ('Shape dims') = 'Shape (Unify [Dim (Name Symbol) (Size Nat)] dims dims')
  Unify [Dim (Name Symbol) (Size Nat)] (dim ': dims) (dim' ': dims') = Unify (Dim (Name Symbol) (Size Nat)) dim dim' ': Unify [Dim (Name Symbol) (Size Nat)] dims dims'
  Unify [Dim (Name Symbol) (Size Nat)] dims dims' = TypeError (UnifyDimsErrorMessage dims dims')
  Unify (Dim (Name Symbol) (Size Nat)) ('Dim name size) ('Dim name' size') = 'Dim (Unify (Name Symbol) name name') (Unify (Size Nat) size size')
  Unify (Name Symbol) 'UncheckedName _ = 'UncheckedName
  Unify (Name Symbol) _ 'UncheckedName = 'UncheckedName
  Unify (Name Symbol) ('Name name) ('Name "*") = 'Name name
  Unify (Name Symbol) ('Name "*") ('Name name) = 'Name name
  Unify (Name Symbol) ('Name name) ('Name name') = TypeError (UnifyNameErrorMessage name name')
  Unify (Size Nat) 'UncheckedSize _ = 'UncheckedSize
  Unify (Size Nat) _ 'UncheckedSize = 'UncheckedSize
  Unify (Size Nat) ('Size size) ('Size size') = TypeError (UnifySizeErrorMessage size size')

type UnifyCheck :: forall k -> k -> k -> Bool
type family UnifyCheck k a b where
  UnifyCheck _ a a = 'True
  UnifyCheck (Gradient RequiresGradient) 'UncheckedGradient _ = 'True
  UnifyCheck (Gradient RequiresGradient) _ 'UncheckedGradient = 'True
  UnifyCheck (Gradient RequiresGradient) ('Gradient requiresGradient) ('Gradient requiresGradient') = 'False
  UnifyCheck (Layout LayoutType) 'UncheckedLayout _ = 'True
  UnifyCheck (Layout LayoutType) _ 'UncheckedLayout = 'True
  UnifyCheck (Layout LayoutType) ('Layout layoutType) ('Layout layoutType') = 'False
  UnifyCheck (Device (DeviceType Nat)) 'UncheckedDevice _ = 'True
  UnifyCheck (Device (DeviceType Nat)) _ 'UncheckedDevice = 'True
  UnifyCheck (Device (DeviceType Nat)) ('Device deviceType) ('Device deviceType') = 'False
  UnifyCheck (DataType DType) 'UncheckedDataType _ = 'True
  UnifyCheck (DataType DType) _ 'UncheckedDataType = 'True
  UnifyCheck (DataType DType) ('DataType dType) ('DataType dType') = 'False
  UnifyCheck (Shape [Dim (Name Symbol) (Size Nat)]) 'UncheckedShape _ = 'True
  UnifyCheck (Shape [Dim (Name Symbol) (Size Nat)]) _ 'UncheckedShape = 'True
  UnifyCheck (Shape [Dim (Name Symbol) (Size Nat)]) ('Shape dims) ('Shape dims') = 'False
  UnifyCheck [Dim (Name Symbol) (Size Nat)] (dim ': dims) (dim' ': dims') = UnifyCheck (Dim (Name Symbol) (Size Nat)) dim dim' && UnifyCheck [Dim (Name Symbol) (Size Nat)] dims dims'
  UnifyCheck [Dim (Name Symbol) (Size Nat)] dims dims' = 'False
  UnifyCheck (Dim (Name Symbol) (Size Nat)) ('Dim name size) ('Dim name' size') = UnifyCheck (Name Symbol) name name' && UnifyCheck (Size Nat) size size'
  UnifyCheck (Name Symbol) 'UncheckedName _ = 'True
  UnifyCheck (Name Symbol) _ 'UncheckedName = 'True
  UnifyCheck (Name Symbol) ('Name name) ('Name "*") = 'True
  UnifyCheck (Name Symbol) ('Name "*") ('Name name) = 'True
  UnifyCheck (Name Symbol) ('Name name) ('Name name') = 'False
  UnifyCheck (Size Nat) 'UncheckedSize _ = 'True
  UnifyCheck (Size Nat) _ 'UncheckedSize = 'True
  UnifyCheck (Size Nat) ('Size size) ('Size size') = 'False

type UnifyRequiresGradientMessage (requiresGradient :: RequiresGradient) (requiresGradient' :: RequiresGradient) =
  "The supplied tensors must all either require or disable gradient calculation,"
    % "but different gradient settings were found:"
    % ""
    % "    " <> requiresGradient <> " and " <> requiresGradient' <> "."
    % ""

type UnifyLayoutErrorMessage (layoutType :: k) (layoutType' :: k') =
  "The supplied tensors must have the same memory layout,"
    % "but different layouts were found:"
    % ""
    % "    " <> layoutType <> " and " <> layoutType' <> "."
    % ""

type UnifyDeviceErrorMessage (deviceType :: k) (deviceType' :: k') =
  "The supplied tensors must be on the same device, "
    % "but different device locations were found:"
    % ""
    % "    " <> deviceType <> " and " <> deviceType' <> "."
    % ""

type UnifyDataTypeErrorMessage (dType :: k) (dType' :: k') =
  "The supplied tensors must have the same data type, "
    % "but different data types were found:"
    % ""
    % "    " <> dType <> " and " <> dType' <> "."
    % ""

type UnifyDimsErrorMessage (dims :: k) (dims' :: k') =
  "The supplied tensors must have shapes with identical number of dimensions,"
    % "but dimension lists of different lengths were found."
    % "Here are the tails of both dimension lists:"
    % ""
    % "    " <> dims <> " and " <> dims' <> "."
    % ""
    % "Try extending, (un-)squeezing, or broadcasting the tensor(s)."

type UnifyNameErrorMessage (name :: k) (name' :: k') =
  "The supplied dimensions must be the same,"
    % "but dimensions with different names were found:"
    % ""
    % "    " <> name <> " and " <> name' <> "."
    % ""
    % "Check spelling and whether or not this is really what you want."
    % "If you are certain, consider dropping or changing the names."

type UnifySizeErrorMessage (size :: k) (size' :: k') =
  "The supplied dimensions must be the same,"
    % "but dimensions with different sizes were found:"
    % ""
    % "    " <> size <> " and " <> size' <> "."
    % ""
    % "Check whether or not this is really what you want."
    % "If you are certain, adjust the sizes such that they match."

type UnifyRightAssociativeL k a b c = Unify k (Unify k a b) c ~ Unify k a (Unify k b c)

type UnifyIdempotenceL2 k a b = Unify k a (Unify k a b) ~ Unify k a b

type UnifyIdempotenceL2C k a b = Unify k a (Unify k b a) ~ Unify k a b

type UnifyIdempotenceL3 k a b c = Unify k a (Unify k b (Unify k a c)) ~ Unify k a (Unify k b c)

type UnifyIdempotenceL3C k a b c = Unify k a (Unify k b (Unify k c a)) ~ Unify k a (Unify k b c)

type UnifyIdempotenceL4 k a b c d = Unify k a (Unify k b (Unify k c (Unify k a d))) ~ Unify k a (Unify k b (Unify k c d))

type UnifyIdempotenceL4C k a b c d = Unify k a (Unify k b (Unify k c (Unify k d a))) ~ Unify k a (Unify k b (Unify k c d))

type UnifyIdempotenceL5 k a b c d e = Unify k a (Unify k b (Unify k c (Unify k d (Unify k a e)))) ~ Unify k a (Unify k b (Unify k c (Unify k d e)))

type UnifyIdempotenceL5C k a b c d e = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e a)))) ~ Unify k a (Unify k b (Unify k c (Unify k d e)))

type UnifyIdempotenceL6 k a b c d e f = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k a f))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e f))))

type UnifyIdempotenceL6C k a b c d e f = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f a))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e f))))

type UnifyIdempotenceL7 k a b c d e f g = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k a g)))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f g)))))

type UnifyIdempotenceL7C k a b c d e f g = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g a)))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f g)))))

type UnifyIdempotenceL8 k a b c d e f g h = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k a h))))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g h))))))

type UnifyIdempotenceL8C k a b c d e f g h = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k h a))))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g h))))))

type UnifyIdempotenceL9 k a b c d e f g h i = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k h (Unify k a i)))))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k h i)))))))

type UnifyIdempotenceL9C k a b c d e f g h i = Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k h (Unify k i a)))))))) ~ Unify k a (Unify k b (Unify k c (Unify k d (Unify k e (Unify k f (Unify k g (Unify k h i)))))))

type (<|>) :: forall k. k -> k -> k
type family (<|>) a b where
  (<|>) (a :: k) (b :: k) = Or k a b

infixr 8 <|>

type Or :: forall k -> k -> k -> k
type family Or k a b where
  Or _ a a = a
  Or (Gradient RequiresGradient) _ ('Gradient 'WithGradient) = 'Gradient 'WithGradient
  Or (Gradient RequiresGradient) 'UncheckedGradient ('Gradient 'WithoutGradient) = 'UncheckedGradient
  Or (Gradient RequiresGradient) ('Gradient 'WithGradient) _ = 'Gradient 'WithGradient
  Or (Gradient RequiresGradient) ('Gradient 'WithoutGradient) 'UncheckedGradient = 'UncheckedGradient

type OrRightAssociativeL k a b c = Or k (Or k a b) c ~ Or k a (Or k b c)

type OrIdempotenceL2 k a b = Or k a (Or k a b) ~ Or k a b

type OrIdempotenceL2C k a b = Or k a (Or k b a) ~ Or k a b

type OrIdempotenceL3 k a b c = Or k a (Or k b (Or k a c)) ~ Or k a (Or k b c)

type OrIdempotenceL3C k a b c = Or k a (Or k b (Or k c a)) ~ Or k a (Or k b c)

type OrIdempotenceL4 k a b c d = Or k a (Or k b (Or k c (Or k a d))) ~ Or k a (Or k b (Or k c d))

type OrIdempotenceL4C k a b c d = Or k a (Or k b (Or k c (Or k d a))) ~ Or k a (Or k b (Or k c d))

type OrIdempotenceL5 k a b c d e = Or k a (Or k b (Or k c (Or k d (Or k a e)))) ~ Or k a (Or k b (Or k c (Or k d e)))

type OrIdempotenceL5C k a b c d e = Or k a (Or k b (Or k c (Or k d (Or k e a)))) ~ Or k a (Or k b (Or k c (Or k d e)))

type OrIdempotenceL6 k a b c d e f = Or k a (Or k b (Or k c (Or k d (Or k e (Or k a f))))) ~ Or k a (Or k b (Or k c (Or k d (Or k e f))))

type OrIdempotenceL6C k a b c d e f = Or k a (Or k b (Or k c (Or k d (Or k e (Or k f a))))) ~ Or k a (Or k b (Or k c (Or k d (Or k e f))))
