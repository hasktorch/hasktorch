{-# LANGUAGE TupleSections #-}
module CodeGen.ParseSpec where

import Test.Hspec
import Text.Megaparsec hiding (runParser')
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

  it "should return all functions in THFile" $ do
    let res = testCGParser thFileContents
    res `shouldSatisfy` isRight
    let Right contents = res
    length contents `shouldSatisfy` (== 155)
    length (catMaybes contents) `shouldSatisfy` (== 70)
    length (nub $ catMaybes contents) `shouldSatisfy` (== 70)


fullGenericParser :: Spec
fullGenericParser = do
  it "should return valid types for the example string" $
    testCGParser exampleGeneric `shouldBe` (Right [Just exampleGeneric'])

  it "should return valid types for the example string with junk" $
    testCGParser (withAllJunk exampleGeneric) `shouldBe` (Right [Nothing, Just exampleGeneric', Nothing, Nothing])

  it "should return valid types for THTensorCopy" $
    pendingWith "need to get THTensorCopy code here"

  it "should return valid types for THNN" $
    pendingWith "we don't currently support THNN"

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
