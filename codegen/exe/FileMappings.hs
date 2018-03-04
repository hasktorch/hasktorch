module FileMappings where

import Data.Monoid ((<>))
import Data.Text
import qualified Data.Text as T

import CodeGenParse (THFunction, Parser, thParseGeneric)
import CodeGen.Types (HModule, TemplateType, genericTypes)
import RenderShared (makeTHModule, renderCHeaderFile, parseFile, IsTemplate(..))

import CLIOptions

thGenericFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
thGenericFiles =
  [ (src <> "THBlas.h"         , (makeGenericModule "THBlas.h" "Blas" "Blas"))
  , (src <> "THLapack.h"       , (makeGenericModule "THLapack.h" "Lapack" "Lapack"))
  , (src <> "THStorage.h"      , (makeGenericModule "THStorage.h" "Storage" "Storage"))
  , (src <> "THStorageCopy.h"  , (makeGenericModule "THStorageCopy.h" "Storage" "StorageCopy"))
  , (src <> "THTensor.h"       , (makeGenericModule "THTensor.h" "Tensor" "Tensor"))
  , (src <> "THTensorConv.h"   , (makeGenericModule "THTensorConv.h" "Tensor" "TensorConv"))
  , (src <> "THTensorCopy.h"   , (makeGenericModule "THTensorCopy.h" "Tensor" "TensorCopy"))
  , (src <> "THTensorLapack.h" , (makeGenericModule "THTensorLapack.h" "Tensor" "TensorLapack"))
  , (src <> "THTensorMath.h"   , (makeGenericModule "THTensorMath.h" "Tensor" "TensorMath"))
  , (src <> "THTensorRandom.h" , (makeGenericModule "THTensorRandom.h" "Tensor" "TensorRandom"))
  , (src <> "THVector.h"       , (makeGenericModule "THVector.h" "Vector" "Vector"))
  ]
 where
  makeGenericModule :: FilePath -> Text -> Text -> (TemplateType -> [THFunction] -> HModule)
  makeGenericModule = makeTHModule out True

  out :: Text
  out = T.pack $ outDir TH GenericFiles

  src :: FilePath
  src = srcDir TH GenericFiles

type HeaderFile = Text

thFiles :: CodeGenType -> [(String, TemplateType -> [THFunction] -> HModule)]
thFiles = \case
  GenericFiles ->
    [ mkTTuple "THBlas.h" "Blas"
    , (src GenericFiles <> "THLapack.h"       , (makeTHModule (out GenericFiles) True "THLapack.h" "Lapack" "Lapack"))
    , (src GenericFiles <> "THStorage.h"      , (makeTHModule (out GenericFiles) True "THStorage.h" "Storage" "Storage"))
    , (src GenericFiles <> "THStorageCopy.h"  , (makeTHModule (out GenericFiles) True "THStorageCopy.h" "Storage" "StorageCopy"))
    , (src GenericFiles <> "THTensor.h"       , (makeTHModule (out GenericFiles) True "THTensor.h" "Tensor" "Tensor"))
    , (src GenericFiles <> "THTensorConv.h"   , (makeTHModule (out GenericFiles) True "THTensorConv.h" "Tensor" "TensorConv"))
    , (src GenericFiles <> "THTensorCopy.h"   , (makeTHModule (out GenericFiles) True "THTensorCopy.h" "Tensor" "TensorCopy"))
    , (src GenericFiles <> "THTensorLapack.h" , (makeTHModule (out GenericFiles) True "THTensorLapack.h" "Tensor" "TensorLapack"))
    , (src GenericFiles <> "THTensorMath.h"   , (makeTHModule (out GenericFiles) True "THTensorMath.h" "Tensor" "TensorMath"))
    , (src GenericFiles <> "THTensorRandom.h" , (makeTHModule (out GenericFiles) True "THTensorRandom.h" "Tensor" "TensorRandom"))
    , (src GenericFiles <> "THVector.h"       , (makeTHModule (out GenericFiles) True "THVector.h" "Vector" "Vector"))
    ]

  ConcreteFiles ->
    [ (src ConcreteFiles <> "THFile.h"        , (makeTHModule (out ConcreteFiles) False "THFile.h" "File" "File"))
    , (src ConcreteFiles <> "THDiskFile.h"    , (makeTHModule (out ConcreteFiles) False "THDiskFile.h" "DiskFile" "DiskFile"))
    , (src ConcreteFiles <> "THAtomic.h"      , (makeTHModule (out ConcreteFiles) False "THDiskFile.h" "Atomic" "Atomic"))
    , (src ConcreteFiles <> "THHalf.h"        , (makeTHModule (out ConcreteFiles) False "THHalf.h" "Half" "Half"))
    , (src ConcreteFiles <> "THLogAdd.h"      , (makeTHModule (out ConcreteFiles) False "THLogAdd.h" "LogAdd" "LogAdd"))
    , (src ConcreteFiles <> "THRandom.h"      , (makeTHModule (out ConcreteFiles) False "THRandom.h" "Random" "Random"))
    , (src ConcreteFiles <> "THSize.h"        , (makeTHModule (out ConcreteFiles) False "THSize.h" "Size" "Size"))
    , (src ConcreteFiles <> "THStorage.h"     , (makeTHModule (out ConcreteFiles) False "THStorage.h" "Storage" "Storage"))
    , (src ConcreteFiles <> "THMemoryFile.h"  , (makeTHModule (out ConcreteFiles) False "THMemoryFile.h" "MemoryFile" "MemoryFile"))
    ]
 where
  out :: CodeGenType -> Text
  out = T.pack . outDir TH

  src :: CodeGenType -> FilePath
  src = srcDir TH

  mkTuple :: IsTemplate -> HeaderFile -> Text -> Text -> CodeGenType -> (String, TemplateType -> [THFunction] -> HModule)
  mkTuple b hf mod cgt = (src cgt <> hf, makeTHModule (out cgt) (isTemplate b) hf mod mod)

  mkTTuple, mkFTuple :: HeaderFile -> Text -> Text -> CodeGenType -> (String, TemplateType -> [THFunction] -> HModule)
  mkTTuple = mkTuple (IsTemplate True)
  mkFTuple = mkTuple (IsTemplate False)
