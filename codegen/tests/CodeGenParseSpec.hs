module CodeGenParseSpec where

import Test.Hspec
import Text.Megaparsec (parseTest)
import CodeGenParse

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "test1" $ parseTest thParseConcrete "TH_API \n"
  it "test2" $ parseTest thParseConcrete "foob TH_API \n"
  it "test3" $ parseTest thParseConcrete "TH_API size_t THFile_readStringRaw(THFile *self, const char *format, char **str_); /* you must deallocate str_ */"
  it "test4" $ parseTest thParseConcrete "TH_API const double THLog2Pi;"
  it "test5" $ parseTest thParseConcrete "TH_API double THLogAdd(double log_a, double log_b);"
