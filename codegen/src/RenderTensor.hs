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
import qualified Data.HashMap.Strict as HM (fromList)


cabal :: Cabal
cabal = Cabal { cabal_pkgname = CabalName "output"
              , cabal_cheaderprefix = "Aten"
              , cabal_moduleprefix = "Aten.Type"
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

--tensorapp :: Types
--tensorapp = cppclass_ t_tensor

t_Generator =
  AbstractClass cabal "Generator" [] mempty Nothing
  []
  []
  []

t_IntArrayRef =
  Class cabal "IntArrayRef" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

t_Scalar =
  Class cabal "Scalar" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

t_ScalarType =
  Class cabal "ScalarType" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

t_SparseTensorRef =
  Class cabal "SparseTensorRef" [] mempty Nothing
  [ Constructor [(cppclassref_ t_Tensor, "t")] Nothing
  ]
  []
  []

t_Storage =
  Class cabal "Storage" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

t_Tensor =
  Class cabal "Tensor" [] mempty Nothing
  [ Constructor [] Nothing
  , NonVirtual long_ "dim" [] Nothing
  , NonVirtual long_ "storage_offset" [] Nothing
  , NonVirtual bool_ "defined" [] Nothing
  , NonVirtual void_ "reset" [] Nothing
  , NonVirtual void_ "cpu" [] Nothing
  , NonVirtual void_ "cuda" [] Nothing
  , NonVirtual void_ "print" [] Nothing
  ]
  []
  []

t_TensorList =
  Class cabal "TensorList" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

t_TensorOptions =
  Class cabal "TensorOptions" [] mempty Nothing
  [ Constructor [] Nothing
  ]
  []
  []

classes :: [Class]
classes =
  [ t_Generator
  , t_IntArrayRef
  , t_Scalar
  , t_ScalarType
  , t_SparseTensorRef
  , t_Storage
  , t_Tensor
  , t_TensorList
  , t_TensorOptions
  ]

toplevelfunctions =  [ ]

templates = []

headerMap :: ModuleUnitMap
headerMap =
  ModuleUnitMap $
    HM.fromList $
      [ ( MU_Class "Generator"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "IntArrayRef"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "Scalar"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "ScalarType"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "SparseTensorRef"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "Storage"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "Tensor"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "TensorList"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      , ( MU_Class "TensorOptions"
        , ModuleUnitImports {
            muimports_namespaces = [NS "at"]
          , muimports_headers = [HdrName "Mask.h",
                                 HdrName "ATen/ATen.h"]
          }
        )
      ]

tensorBuilder :: IO ()
tensorBuilder = do
  simpleBuilder "Aten.Type" headerMap (cabal,classes,toplevelfunctions,templates)
    [ ] extraDep


