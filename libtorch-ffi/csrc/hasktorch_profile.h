// when setting --enable-profiling, PROFILING is defined by GHC. torch uses PROFILING literal. To avoid the confliction, PROFILING is redefined to TORCH_PROFILING.
#ifdef PROFILING
#undef PROFILING
#define PROFILING TORCH_PROFILING
#endif
