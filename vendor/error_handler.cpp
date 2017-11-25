#include <iostream>
#include <sstream>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <typeinfo>

#include "error_handler.h"

extern "C" void errorHandler(const char *format, ...);
extern "C" void argErrorHandler(int arg, const char * msg, void * data);
extern "C" void testFunction();

// errorHandler is cast as THErrorHandlerFunction, see THGeneral.h.in
// typedef void (*THErrorHandlerFunction)(const char *msg, void *data);
// void runtime_error(const char *format, ...) {
void errorHandler(const char *format, ...) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];

  std::cout << "--- custom error handler ---" << std::endl;

  std::va_list fmt_args;
  va_start(fmt_args, format);
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);
  va_end(fmt_args);

  throw std::runtime_error(error_buf);
}

void argErrorHandler(int arg, const char * msg, void * data) {
  std::stringstream new_error;
  new_error << "invalid argument " << arg << ": " << msg;
  throw std::runtime_error(new_error.str());
}

void testFunction() {
  std::cout << "Test function" << std::endl;
}
