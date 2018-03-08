{-# LANGUAGE TupleSections #-}
module CodeGen.ParseSpec where

import Test.Hspec
import Text.Megaparsec hiding (runParser', State)
import CodeGen.Parse hiding (describe)
import CodeGen.Types hiding (describe)
import CodeGen.Prelude
import Data.List
import Data.Either
import Data.Maybe
import qualified Data.Text as T


main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "the skip parser" skipSpec
  describe "the ptr parser" ptrSpec
  describe "the ptr2 parser" ptr2Spec
  describe "the ctypes parser" ctypesSpec
  describe "the args parser" argsSpec
  describe "the function parser" functionSpec
  describe "running the full parser" $ do
    describe "in concrete mode" fullConcreteParser
    describe "in generic mode" fullGenericParser

runParser' p = runParser p "test"

testCGParser = runParser parser "test"

succeeds :: (Show x, Eq x) => Parser x -> (String, x) -> IO ()
succeeds parser (str, res) = runParser' parser str `shouldBe` (Right res)

fails :: (Show x, Eq x) => Parser x -> String -> IO ()
fails parser str = runParser' parser str `shouldSatisfy` isLeft

fullConcreteParser :: Spec
fullConcreteParser = do
  it "should error if the header is first(??)" $
    testCGParser "TH_API \n" `shouldBe` (Right [Nothing])

  it "should return Nothing if the API header" $
    testCGParser "foob TH_API \n" `shouldBe` (Right [Nothing])

  it "should return a valid THFile function" $ do
    testCGParser thFileFunction `shouldBe` (Right [Just thFileFunctionRendered])

  it "should return Nothing for declarations" $
    testCGParser "TH_API const double THLog2Pi;" `shouldBe` (Right [Nothing])

  it "should return valid THLogAdd functions" $
    testCGParser thLogAddFunction `shouldBe` (Right [Just thLogAddFunctionRendered])

  it "should return all functions in THFile" $
    thFileContents `fullFileTest` (155, 70, 70)


fullGenericParser :: Spec
fullGenericParser = do
  it "should return valid types for the example string" $
    testCGParser exampleGeneric `shouldBe` (Right [Just exampleGeneric'])

  it "should return valid types for the example string with junk" $
    testCGParser (withAllJunk exampleGeneric) `shouldBe` (Right [Nothing, Just exampleGeneric', Nothing, Nothing])

  it "should return all functions for THStorage" $
    thGenericStorageContents `fullFileTest` (86, 22, 22)

  it "should return valid types for THNN" $
    pendingWith "we don't currently support THNN"

fullFileTest :: String -> (Int, Int, Int) -> Expectation
fullFileTest contents (all, found, uniq) = do
  let res = testCGParser contents
  res `shouldSatisfy` isRight
  let Right contents = res
  length contents `shouldSatisfy` (== all)
  length (catMaybes contents) `shouldSatisfy` (== found)
  length (nub $ catMaybes contents) `shouldSatisfy` (== uniq)


skipSpec :: Spec
skipSpec = do
  it "finds the number of lines as \\n characters" $ do
    runSomeSkipOn "\n\n" `shouldSatisfy` (foundNlines 2)
  it "finds the number of lines in exampleGeneric" $ do
    runSomeSkipOn exampleGeneric `shouldSatisfy` (foundNlines 1)
  it "finds the number of lines in exampleGeneric withEndJunk" $ do
    runSomeSkipOn (withEndJunk exampleGeneric) `shouldSatisfy` (foundNlines 2)
  it "finds the number of lines in exampleGeneric withStartJunk" $ do
    runSomeSkipOn (withStartJunk exampleGeneric) `shouldSatisfy` (foundNlines 2)
  it "finds the number of lines in exampleGeneric withAllJunk" $ do
    runSomeSkipOn (withAllJunk exampleGeneric) `shouldSatisfy` (foundNlines 3)
 where
  foundNlines n = either (const False) ((== n) . length)
  runSomeSkipOn = runParser' (some skip)

ptrSpec :: Spec
ptrSpec = do
  it "will parse pointers correctly"         $ mapM_ (succeeds ptr . (,())) [" * ", " *", "* ", "*"]
  it "will parse invalid pointers correctly" $ mapM_ (fails ptr)    [" ", ";*", "_* ", ""]

ptr2Spec :: Spec
ptr2Spec = do
  it "will not parse single pointers" $ mapM_ (fails ptr2) [" * ", " *", "* ", "*"]
  it "will not parse invalid inputs"  $ mapM_ (fails ptr2) [" ", ";*", "_* ", ""]
  it "will parse double pointers" $ mapM_ (succeeds ptr2 . (,()))
    [ " ** " , " **" , "** " , "**" , " * * " , " * *" , "* * " , "* *"
    , " * * " , " * *" , "* * " , "* *", "**      ", "     ** ", "  *   * "
    ]

ctypesSpec :: Spec
ctypesSpec = do
  describe "direct types" $ do
    it "renders happy path CTypes" $
      mapM_ (succeeds ctypes)
        [ ("uint64_t", CType CUInt64)
        , ("int", CType CInt)
        , ("int64_t", CType CInt64)
        , ("void", CType CVoid)
        ]

  describe "pointer ctypes" $ do
    it "renders happy path CType pointers" $
      mapM_ (succeeds ctypes)
        [ ("uint64_t *", Ptr (CType CUInt64))
        , ("int*", Ptr (CType CInt))
        , ("int64_t *", Ptr (CType CInt64))
        , ("void   *   ", Ptr (CType CVoid))
        ]

  describe "double-pointer ctypes" $ do
    it "renders happy path CType pointers of pointers" $
      mapM_ (succeeds ctypes)
        [ ("uint64_t * *", Ptr (Ptr (CType CUInt64)))
        , ("int**", Ptr (Ptr (CType CInt)))
        , ("int64_t **", Ptr (Ptr (CType CInt64)))
        , ("void   *   * ", Ptr (Ptr (CType CVoid)))
        ]
argsSpec :: Spec
argsSpec = do
  it "will find arguments with no name" $
    runParser' (char '(' >> functionArg) "(void)" `shouldBe` (Right (Arg (CType CVoid) ""))

functionSpec :: Spec
functionSpec = do
  it "will find functions where the arguments have no name" $
    runParser' function storageElementSize `shouldBe` (Right (Just storageElementSize'))

-- ========================================================================= --


exampleGeneric :: String
exampleGeneric = "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);"
exampleGeneric' = Function "setFlag" [ Arg (Ptr (TenType Tensor)) "self", Arg (CType CChar) "flag"] (CType CVoid)

withStartJunk :: String -> String
withStartJunk x = "skip this garbage line line\n" <> x

withEndJunk :: String -> String
withEndJunk x = x <> "\nanother garbage line ( )@#R @# 324 32"

withAllJunk :: String -> String
withAllJunk = withEndJunk . withStartJunk

thFileFunction = intercalate ""
  [ "TH_API size_t THFile_readStringRaw(THFile *self, const char *format, char **str_); /* you must "
  , "deallocate str_ */"
  ]

thFileFunctionRendered = Function "THFile_readStringRaw"
  [ Arg      (Ptr (TenType File)) "self"
  , Arg      (Ptr (CType CChar)) "format"
  , Arg (Ptr (Ptr (CType CChar))) "str_"
  ] (CType CSize)

thLogAddFunction :: String
thLogAddFunction = "TH_API double THLogAdd(double log_a, double log_b);"
thLogAddFunctionRendered = Function "THLogAdd"
  [ Arg (CType CDouble) "log_a"
  , Arg (CType CDouble) "log_b"
  ] (CType CDouble)

-- do this later
thNNFunction = "TH_API void THNN_(Abs_updateOutput)(THNNState *state, THTensor *input, THTensor *output);"
thNNFunctionRendered = undefined

thFileContents = intercalate ""
  [ "#ifndef TH_FILE_INC\n#define TH_FILE_INC\n\n#include \"THStorage.h\"\n\ntypedef struct THFile__ "
  , "THFile;\n\nTH_API int THFile_isOpened(THFile *self);\nTH_API int THFile_isQuiet(THFile *self);\n"
  , "TH_API int THFile_isReadable(THFile *self);\nTH_API int THFile_isWritable(THFile *self);\nTH_API"
  , "int THFile_isBinary(THFile *self);\nTH_API int THFile_isAutoSpacing(THFile *self);\nTH_API int T"
  , "HFile_hasError(THFile *self);\n\nTH_API void THFile_binary(THFile *self);\nTH_API void THFile_as"
  , "cii(THFile *self);\nTH_API void THFile_autoSpacing(THFile *self);\nTH_API void THFile_noAutoSpac"
  , "ing(THFile *self);\nTH_API void THFile_quiet(THFile *self);\nTH_API void THFile_pedantic(THFile "
  , "*self);\nTH_API void THFile_clearError(THFile *self);\n\n/* scalar */\nTH_API uint8_t THFile_rea"
  , "dByteScalar(THFile *self);\nTH_API int8_t THFile_readCharScalar(THFile *self);\nTH_API int16_t T"
  , "HFile_readShortScalar(THFile *self);\nTH_API int32_t THFile_readIntScalar(THFile *self);\nTH_API"
  , "int64_t THFile_readLongScalar(THFile *self);\nTH_API float THFile_readFloatScalar(THFile *self);"
  , "\nTH_API double THFile_readDoubleScalar(THFile *self);\n\nTH_API void THFile_writeByteScalar(THF"
  , "ile *self, uint8_t scalar);\nTH_API void THFile_writeCharScalar(THFile *self, int8_t scalar);\nT"
  , "H_API void THFile_writeShortScalar(THFile *self, int16_t scalar);\nTH_API void THFile_writeIntSc"
  , "alar(THFile *self, int32_t scalar);\nTH_API void THFile_writeLongScalar(THFile *self, int64_t sc"
  , "alar);\nTH_API void THFile_writeFloatScalar(THFile *self, float scalar);\nTH_API void THFile_wri"
  , "teDoubleScalar(THFile *self, double scalar);\n\n/* storage */\nTH_API size_t THFile_readByte(THF"
  , "ile *self, THByteStorage *storage);\nTH_API size_t THFile_readChar(THFile *self, THCharStorage *"
  , "storage);\nTH_API size_t THFile_readShort(THFile *self, THShortStorage *storage);\nTH_API size_t"
  , "THFile_readInt(THFile *self, THIntStorage *storage);\nTH_API size_t THFile_readLong(THFile *self"
  , ", THLongStorage *storage);\nTH_API size_t THFile_readFloat(THFile *self, THFloatStorage *storage"
  , ");\nTH_API size_t THFile_readDouble(THFile *self, THDoubleStorage *storage);\n\nTH_API size_t TH"
  , "File_writeByte(THFile *self, THByteStorage *storage);\nTH_API size_t THFile_writeChar(THFile *se"
  , "lf, THCharStorage *storage);\nTH_API size_t THFile_writeShort(THFile *self, THShortStorage *stor"
  , "age);\nTH_API size_t THFile_writeInt(THFile *self, THIntStorage *storage);\nTH_API size_t THFile"
  , "_writeLong(THFile *self, THLongStorage *storage);\nTH_API size_t THFile_writeFloat(THFile *self,"
  , "THFloatStorage *storage);\nTH_API size_t THFile_writeDouble(THFile *self, THDoubleStorage *stora"
  , "ge);\n\n/* raw */\nTH_API size_t THFile_readByteRaw(THFile *self, uint8_t *data, size_t n);\nTH_"
  , "API size_t THFile_readCharRaw(THFile *self, int8_t *data, size_t n);\nTH_API size_t THFile_readS"
  , "hortRaw(THFile *self, int16_t *data, size_t n);\nTH_API size_t THFile_readIntRaw(THFile *self, i"
  , "nt32_t *data, size_t n);\nTH_API size_t THFile_readLongRaw(THFile *self, int64_t *data, size_t n"
  , ");\nTH_API size_t THFile_readFloatRaw(THFile *self, float *data, size_t n);\nTH_API size_t THFil"
  , "e_readDoubleRaw(THFile *self, double *data, size_t n);\nTH_API size_t THFile_readStringRaw(THFil"
  , "e *self, const char *format, char **str_); /* you must deallocate str_ */\n\nTH_API size_t THFil"
  , "e_writeByteRaw(THFile *self, uint8_t *data, size_t n);\nTH_API size_t THFile_writeCharRaw(THFile"
  , "*self, int8_t *data, size_t n);\nTH_API size_t THFile_writeShortRaw(THFile *self, int16_t *data,"
  , "size_t n);\nTH_API size_t THFile_writeIntRaw(THFile *self, int32_t *data, size_t n);\nTH_API siz"
  , "e_t THFile_writeLongRaw(THFile *self, int64_t *data, size_t n);\nTH_API size_t THFile_writeFloat"
  , "Raw(THFile *self, float *data, size_t n);\nTH_API size_t THFile_writeDoubleRaw(THFile *self, dou"
  , "ble *data, size_t n);\nTH_API size_t THFile_writeStringRaw(THFile *self, const char *str, size_t"
  , "size);\n\nTH_API THHalf THFile_readHalfScalar(THFile *self);\nTH_API void THFile_writeHalfScalar"
  , "(THFile *self, THHalf scalar);\nTH_API size_t THFile_readHalf(THFile *self, THHalfStorage *stora"
  , "ge);\nTH_API size_t THFile_writeHalf(THFile *self, THHalfStorage *storage);\nTH_API size_t THFil"
  , "e_readHalfRaw(THFile *self, THHalf* data, size_t size);\nTH_API size_t THFile_writeHalfRaw(THFil"
  , "e *self, THHalf* data, size_t size);\n\nTH_API void THFile_synchronize(THFile *self);\nTH_API vo"
  , "id THFile_seek(THFile *self, size_t position);\nTH_API void THFile_seekEnd(THFile *self);\nTH_AP"
  , "I size_t THFile_position(THFile *self);\nTH_API void THFile_close(THFile *self);\nTH_API void TH"
  , "File_free(THFile *self);\n\n#endif\n"
  ]

storageElementSize :: String
storageElementSize = "TH_API size_t THStorage_(elementSize)(void);"

storageElementSize' :: Function
storageElementSize' = Function "elementSize" [ Arg (CType CVoid) "" ] (CType CSize)

thGenericStorageContents :: String
thGenericStorageContents = intercalate ""
  [ "#ifndef TH_GENERIC_FILE\n#define TH_GENERIC_FILE \"generic/THStorage.h\"\n#else\n\n/* on pourra"
  , "it avoir un liste chainee\n   qui initialise math, lab structures (or more).\n   mouais -- comp"
  , "lique.\n\n   Pb: THMapStorage is kind of a class\n   THLab_()... comment je m'en sors?\n\n   en"
  , " template, faudrait que je les instancie toutes!!! oh boy!\n   Et comment je sais que c'est pou"
  , "r Cuda? Le type float est le meme dans les <>\n\n   au bout du compte, ca serait sur des pointe"
  , "urs float/double... etc... = facile.\n   primitives??\n */\n\n#define TH_STORAGE_REFCOUNTED 1\n"
  , "#define TH_STORAGE_RESIZABLE  2\n#define TH_STORAGE_FREEMEM    4\n#define TH_STORAGE_VIEW      "
  , " 8\n\ntypedef struct THStorage\n{\n    real *data;\n    ptrdiff_t size;\n    int refcount;\n   "
  , " char flag;\n    THAllocator *allocator;\n    void *allocatorContext;\n    struct THStorage *vi"
  , "ew;\n} THStorage;\n\nTH_API real* THStorage_(data)(const THStorage*);\nTH_API ptrdiff_t THStora"
  , "ge_(size)(const THStorage*);\nTH_API size_t THStorage_(elementSize)(void);\n\n/* slow access --"
  , " checks everything */\nTH_API void THStorage_(set)(THStorage*, ptrdiff_t, real);\nTH_API real T"
  , "HStorage_(get)(const THStorage*, ptrdiff_t);\n\nTH_API THStorage* THStorage_(new)(void);\nTH_AP"
  , "I THStorage* THStorage_(newWithSize)(ptrdiff_t size);\nTH_API THStorage* THStorage_(newWithSize"
  , "1)(real);\nTH_API THStorage* THStorage_(newWithSize2)(real, real);\nTH_API THStorage* THStorage"
  , "_(newWithSize3)(real, real, real);\nTH_API THStorage* THStorage_(newWithSize4)(real, real, real"
  , ", real);\nTH_API THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, in"
  , "t flags);\n\n/* takes ownership of data */\nTH_API THStorage* THStorage_(newWithData)(real *dat"
  , "a, ptrdiff_t size);\n\nTH_API THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,\n        "
  , "                                       THAllocator* allocator,\n                               "
  , "                void *allocatorContext);\nTH_API THStorage* THStorage_(newWithDataAndAllocator)"
  , "(\n    real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);\n\n/* shoul"
  , "d not differ with API */\nTH_API void THStorage_(setFlag)(THStorage *storage, const char flag);"
  , "\nTH_API void THStorage_(clearFlag)(THStorage *storage, const char flag);\nTH_API void THStorag"
  , "e_(retain)(THStorage *storage);\nTH_API void THStorage_(swap)(THStorage *storage1, THStorage *s"
  , "torage2);\n\n/* might differ with other API (like CUDA) */\nTH_API void THStorage_(free)(THStor"
  , "age *storage);\nTH_API void THStorage_(resize)(THStorage *storage, ptrdiff_t size);\nTH_API voi"
  , "d THStorage_(fill)(THStorage *storage, real value);\n\n#endif\n"
  ]
