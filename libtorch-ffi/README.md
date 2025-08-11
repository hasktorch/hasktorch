# libtorch-ffi

This package provides FFI bindings to PyTorch's libtorch C++ library.

## Setup

The package automatically downloads and configures libtorch during the build process. You can customize the setup using environment variables.

### Environment Variables

#### `LIBTORCH_VERSION`
- **Default**: `2.5.0`
- **Description**: Specifies the version of libtorch to download and use
- **Example**: `export LIBTORCH_VERSION=2.5.0`

#### `LIBTORCH_HOME`
- **Default**: XDG cache directory (`~/.cache/libtorch` on Linux/macOS)
- **Description**: Base directory where libtorch will be downloaded and stored
- **Example**: `export LIBTORCH_HOME=/opt/libtorch`

#### `LIBTORCH_CUDA_VERSION`
- **Default**: `cpu`
- **Description**: CUDA version for GPU support
- **Options**: 
  - `cpu` - CPU-only version (default)
  - `cu117` - CUDA 11.7
  - `cu118` - CUDA 11.8
  - `cu121` - CUDA 12.1
  - Any other CUDA version string supported by PyTorch
- **Example**: `export LIBTORCH_CUDA_VERSION=cu118`

#### `LIBTORCH_SKIP_DOWNLOAD`
- **Default**: Not set
- **Description**: When set (to any value), skips the automatic download of libtorch
- **Use case**: When you have libtorch already installed system-wide
- **Example**: `export LIBTORCH_SKIP_DOWNLOAD=1`

### Directory Structure

The downloaded libtorch is stored in a platform-specific directory structure:
```
$LIBTORCH_HOME/
└── <version>/
    └── <platform>/
        └── <cuda-flavor>/
            ├── lib/
            ├── include/
            └── .ok
```

Where:
- `<version>` is the libtorch version (e.g., `2.5.0`)
- `<platform>` is one of:
  - `macos-arm64` - macOS on Apple Silicon
  - `macos-x86_64` - macOS on Intel
  - `linux-x86_64` - Linux on x86_64
- `<cuda-flavor>` is the CUDA version (e.g., `cpu`, `cu118`)

### Build Process

1. **Pre-configuration**: The package checks if it's running in a Nix sandbox. If not, it proceeds with the download process.

2. **Download**: If libtorch is not found in the cache directory, it will be automatically downloaded from PyTorch's official servers.

3. **Configuration**: The build system automatically:
   - Adds the libtorch library directory to the library search path
   - Adds the include directories for C++ headers
   - Sets up proper runtime library paths (rpath) for dynamic linking
   - On macOS, adds the `-ld_classic` flag for compatibility

### Platform-Specific Notes

#### macOS
- Uses rpath for dynamic library loading
- Automatically adds `-ld_classic` flag for linker compatibility
- Supports both Apple Silicon (arm64) and Intel (x86_64) architectures
- **Since libtorch-ffi's rpath is propagated, it doesn't matter whether hasktorch is a static link or a shared link**

#### Linux
- Uses rpath for dynamic library loading
- Supports x86_64 architecture
- Multiple CUDA versions available for GPU support
- **Since libtorch-ffi's rpath is not propagated, hasktorch must be a shared link**

### Linking Configuration

Due to rpath propagation differences between platforms, Linux requires shared linking. Add the following configuration:

#### For Cabal (cabal.project)
```
shared: True
executable-dynamic: True
```

#### For Stack (stack.yaml)
```yaml
configure-options:
  $targets:
    - --enable-executable-dynamic
    - --enable-shared
```

### Nix Support

The package detects when it's being built in a Nix sandbox and skips the automatic download. In this case, libtorch should be provided through Nix derivation inputs.

### Troubleshooting

1. **Download failures**: Check your internet connection and ensure the PyTorch download servers are accessible.

2. **Missing libraries**: The `.ok` marker file indicates a successful download. If this file is missing but the directory exists, delete the directory and let the setup download again.

3. **CUDA version mismatch**: Ensure your system CUDA version matches the `LIBTORCH_CUDA_VERSION` you've specified.

4. **Custom libtorch installation**: Set `LIBTORCH_SKIP_DOWNLOAD=1` and ensure your system's libtorch is properly configured in your build environment.

### Example Usage

```bash
# Use CPU-only version
cabal build libtorch-ffi

# Use CUDA 11.8 version
export LIBTORCH_CUDA_VERSION=cu118
cabal build libtorch-ffi

# Use a specific version
export LIBTORCH_VERSION=2.4.0
cabal build libtorch-ffi

# Use existing system libtorch
export LIBTORCH_SKIP_DOWNLOAD=1
cabal build libtorch-ffi
```
