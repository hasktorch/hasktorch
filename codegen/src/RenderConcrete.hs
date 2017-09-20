{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared

parseFilesNoTemplate :: [(String, TemplateType -> [THFunction] -> HModule)]
parseFilesNoTemplate =
  [
    ("vendor/torch7/lib/TH/THFile.h",
     (makeModule "THFile.h" "File" "File")),
    ("vendor/torch7/lib/TH/THHDiskFile.h",
     (makeModule "THDiskFile.h" "Vector" "Vector"))
  ]

main = do
  putStrLn "Done"
