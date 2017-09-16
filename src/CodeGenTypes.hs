{-# LANGUAGE OverloadedStrings #-}

module CodeGenTypes (
  genTypes,
  type2SpliceReal,
  type2real,
  type2accreal,
  realtype2Haskell,
  accrealtype2Haskell,
  TemplateType(..)
  ) where

import Data.Text

data TemplateType = GenByte
                  | GenChar
                  | GenDouble
                  | GenFloat
                  | GenHalf
                  | GenInt
                  | GenLong
                  | GenShort deriving Show

-- List used to iterate through all template types
genTypes :: [TemplateType]
genTypes = [GenByte, GenChar,
            GenDouble, GenFloat, GenHalf,
            GenInt, GenLong, GenShort]

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal GenByte   = "Byte"
type2SpliceReal GenChar   = "Byte"
type2SpliceReal GenDouble = "Double"
type2SpliceReal GenFloat  = "Float"
type2SpliceReal GenHalf   = "Half"
type2SpliceReal GenInt    = "Int"
type2SpliceReal GenLong   = "Long"
type2SpliceReal GenShort  = "Short"

-- #define real [X]
type2real :: TemplateType -> Text
type2real GenByte   = "unsigned char"
type2real GenChar   = "char"
type2real GenDouble = "double"
type2real GenFloat  = "float"
type2real GenHalf   = "THHalf"
type2real GenInt    = "int"
type2real GenLong   = "long"
type2real GenShort  = "short"

-- #define accreal [X]
type2accreal :: TemplateType -> Text
type2accreal GenByte   = "long"
type2accreal GenChar   = "long"
type2accreal GenDouble = "double"
type2accreal GenFloat  = "double"
type2accreal GenHalf   = "float"
type2accreal GenInt    = "long"
type2accreal GenLong   = "long"
type2accreal GenShort  = "long"

realtype2Haskell :: TemplateType -> Text
realtype2Haskell GenByte   = "CUChar"
realtype2Haskell GenChar   = "CChar"
realtype2Haskell GenDouble = "CDouble"
realtype2Haskell GenFloat  = "CFloat"
realtype2Haskell GenHalf   = "THHalf"
realtype2Haskell GenInt    = "CInt"
realtype2Haskell GenLong   = "CLong"
realtype2Haskell GenShort  = "CShort"

accrealtype2Haskell :: TemplateType -> Text
accrealtype2Haskell GenByte   = "CLong"
accrealtype2Haskell GenChar   = "CLong"
accrealtype2Haskell GenDouble = "CDouble"
accrealtype2Haskell GenFloat  = "CDouble"
accrealtype2Haskell GenHalf   = "CFloat"
accrealtype2Haskell GenInt    = "CLong"
accrealtype2Haskell GenLong   = "CLong"
accrealtype2Haskell GenShort  = "CLong"
