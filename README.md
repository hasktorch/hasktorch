# ffi-experimental

Experimental next-gen code generation for aten bindings in preparation for 0.0.2 which targets the 1.0 aten backend.

Ideas being explored:

- Use yaml specs (which seemed to have been cleaned up since PT ~ 0.4) instead of header parsing.
- Try inline-cpp functionality to bind the C++ API instead of the C API. Benchmark potential template haskell overhead vs. other approaches.
- Get a vertical slice working for a small number of functions.
- Scale up.

Contributions/PRs are welcome.