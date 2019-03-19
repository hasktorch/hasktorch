{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderTensor where

import Data.Monoid (mempty)
--
import FFICXX.Generate.Builder
import FFICXX.Generate.Code.Primitive
import FFICXX.Generate.Type.Cabal
import FFICXX.Generate.Type.Class
import FFICXX.Generate.Type.Config
import FFICXX.Generate.Type.Module
import FFICXX.Generate.Type.PackageInterface


cabal = Cabal { cabal_pkgname = CabalName "aten-tensor"
              , cabal_cheaderprefix = "Aten"
              , cabal_moduleprefix = "Aten.Tensor"
              , cabal_version = ">=1.10"
              , cabal_license = Just "BSD3"
              , cabal_additional_c_incs = []
              , cabal_additional_c_srcs = []
              , cabal_additional_pkgdeps = []
              , cabal_licensefile = Nothing
              , cabal_extraincludedirs = []
              , cabal_extralibdirs = []
              , cabal_extrafiles = []
              , cabal_pkg_config_depends = []
              }

extraDep = [] -- [ ("Tensor", ["STL.Vector.Template"]) ]

{-
cabalattr =
    CabalAttr
    { cabalattr_license = Just "BSD3"
    , cabalattr_licensefile = Just "LICENSE"
    , cabalattr_extraincludedirs = []
    , cabalattr_extralibdirs = []
    , cabalattr_extrafiles = []
    }
-}

tensorapp = cppclass_ t_tensor

t_tensor :: Class
t_tensor =
  Class cabal "Tensor" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

classes = [ t_tensor ]

toplevelfunctions =  [ ]

templates = []

headerMap = ModuleUnitMap mempty

tensorBuilder :: IO ()
tensorBuilder = do
  simpleBuilder "Aten.Tensor" headerMap (cabal,classes,toplevelfunctions,templates)
    [ ] extraDep


