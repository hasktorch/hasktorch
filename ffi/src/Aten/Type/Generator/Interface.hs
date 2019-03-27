{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.Generator.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.Generator.RawType

class () => IGenerator a where
