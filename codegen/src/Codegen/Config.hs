{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Codegen.Config where

import Data.Yaml
import GHC.Generics

data CppClassSpec = CppClassSpec
  { signature :: String,
    cppname :: String,
    hsname :: String
  }
  deriving (Show, Eq, Generic)

data CppClassesSpec = CppClassesSpec
  { classes :: [CppClassSpec]
  }
  deriving (Show, Eq, Generic)

cppClassList :: CppClassesSpec
cppClassList =
  CppClassesSpec
  [ CppClassSpec
    { signature = "IntArray"
    , cppname = "std::vector<int64_t>"
    , hsname = "IntArray"
    }
  , CppClassSpec
    { signature = "Int8Array"
    , cppname = "std::vector<uint8_t>"
    , hsname = "Int8Array"
    }
  , CppClassSpec
    { signature = "StringArray"
    , cppname = "std::vector<std::string>"
    , hsname = "StringArray"
    }
  , CppClassSpec
    { signature = "StdMSec"
    , cppname = "std::chrono::milliseconds"
    , hsname = "StdMSec"
    }
  , CppClassSpec
    { signature = "ProcessGroup"
    , cppname = "c10d::ProcessGroup"
    , hsname = "ProcessGroup"
    }
  , CppClassSpec
    { signature = "Work"
    , cppname = "c10::intrusive_ptr<ProcessGroup::Work>"
    , hsname = "Work"
    }
  , CppClassSpec
    { signature = "Store"
    , cppname = "c10d::Store"
    , hsname = "Store"
    }
  , CppClassSpec
    { signature = "BroadcastOptions"
    , cppname = "c10d::BroadcastOptions"
    , hsname = "BroadcastOptions"
    }
  ]

