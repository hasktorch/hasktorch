#include "TH/THStorage.h"
#include "TH/THTensor.h"

// Byte types
//================================================
void free_ByteStorage(void* s, THByteStorage * x) {
  THByteStorage_free(&x);
};

void free_ByteTensor(void* s, THByteTensor * x) {
  THByteTensor_free(&x);
};

// Char types
//================================================
void free_CharStorage(void* s, THCharStorage * x) {
  THCharStorage_free(&x);
};

void free_CharTensor(void* s, THCharTensor * x) {
  THCharTensor_free(&x);
};

// Int types
//================================================
void free_IntStorage(void* s, THIntStorage * x) {
  THIntStorage_free(&x);
};

void free_IntTensor(void* s, THIntTensor * x) {
  THIntTensor_free(&x);
};

// Short types
//================================================
void free_ShortStorage(void* s, THShortStorage * x) {
  THShortStorage_free(&x);
};

void free_ShortTensor(void* s, THShortTensor * x) {
  THShortTensor_free(&x);
};

// Long types
//================================================
void free_LongStorage(void* s, THLongStorage * x) {
  THLongStorage_free(&x);
};

void free_LongTensor(void* s, THLongTensor * x) {
  THLongTensor_free(&x);
};

// Half types
//================================================
void free_HalfStorage(void* s, THHalfStorage * x) {
  THHalfStorage_free(&x);
};

void free_HalfTensor(void* s, THHalfTensor * x) {
  THHalfTensor_free(&x);
};

// Float types
//================================================
void free_FloatStorage(void* s, THFloatStorage * x) {
  THFloatStorage_free(&x);
};

void free_FloatTensor(void* s, THFloatTensor * x) {
  THFloatTensor_free(&x);
};

// Double types
//================================================
void free_DoubleStorage(void* s, THDoubleStorage * x) {
  THDoubleStorage_free(&x);
};

void free_DoubleTensor(void* s, THDoubleTensor * x) {
  THDoubleTensor_free(&x);
};


