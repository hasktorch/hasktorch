#!/usr/bin/env python3
"""
Verify that NVIDIA wheel packages are available on PyPI for a given CUDA version.
This script:
1. Downloads the libtorch package for the specified CUDA version
2. Checks what NVIDIA libraries are missing or needed
3. Verifies those libraries are available on PyPI

Usage: python3 verify-nvidia-wheels.py [cu130|cu121|cu118|...]
"""

import sys
import json
import urllib.request
import urllib.error
import tempfile
import zipfile
import os
import re
from pathlib import Path

LIBTORCH_VERSION = "2.9.1"

def get_libtorch_url(cuda_flavor):
    """Get the libtorch download URL for the given CUDA flavor.

    Always returns Linux URL since that's where the NVIDIA library issue exists.
    macOS builds bundle everything; Linux builds need separate NVIDIA libs.
    """
    if cuda_flavor == "cpu":
        return f"https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-{LIBTORCH_VERSION}%2Bcpu.zip"
    else:
        # Always use Linux build for checking CUDA dependencies
        return f"https://download.pytorch.org/libtorch/{cuda_flavor}/libtorch-shared-with-deps-{LIBTORCH_VERSION}%2B{cuda_flavor}.zip"

def download_and_extract_libtorch(cuda_flavor, tmpdir):
    """Download libtorch and extract it to tmpdir."""
    url = get_libtorch_url(cuda_flavor)
    print(f"Downloading libtorch from: {url}")

    zip_path = os.path.join(tmpdir, "libtorch.zip")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

        print("\n  Download complete. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        lib_dir = os.path.join(tmpdir, "libtorch", "lib")
        if os.path.exists(lib_dir):
            return lib_dir
        else:
            print(f"  ✗ Error: lib directory not found in extracted libtorch")
            return None

    except Exception as e:
        print(f"\n  ✗ Error downloading/extracting libtorch: {e}")
        return None

def extract_soname_dependencies(lib_dir):
    """Extract SONAME dependencies from libtorch libraries using readelf.

    Returns dict mapping library basename (e.g., 'cusparse') to version (e.g., '12')
    """
    import subprocess

    print("\n1. Extracting SONAME dependencies from libtorch libraries:")

    # Check key libtorch libraries
    so_files = ['libtorch_cuda.so', 'libtorch.so', 'libtorch_cpu.so']
    nvidia_deps = {}  # e.g., {'cusparse': '12', 'cufft': '11', ...}

    for so_file in so_files:
        full_path = os.path.join(lib_dir, so_file)
        if not os.path.exists(full_path):
            continue

        print(f"\n  Checking {so_file}:")

        try:
            # Use readelf to get NEEDED entries (works on Linux binaries from any platform)
            result = subprocess.run(
                ['readelf', '-d', full_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                print(f"    ⚠ readelf failed, trying objdump...")
                # Fallback to objdump
                result = subprocess.run(
                    ['objdump', '-p', full_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            # Parse output for NVIDIA library dependencies
            # Look for patterns like: libcusparse.so.12, libcufft.so.11
            for line in result.stdout.split('\n'):
                if 'NEEDED' in line or 'libcu' in line.lower():
                    # Extract library names like "libcusparse.so.12"
                    match = re.search(r'lib(cusparse|cufft|curand|cublas|nvjitlink|cusolver)\.so\.(\d+)', line)
                    if match:
                        lib_name = match.group(1)
                        version = match.group(2)

                        if lib_name not in nvidia_deps:
                            nvidia_deps[lib_name] = version
                            print(f"    ✓ Needs lib{lib_name}.so.{version}")

        except FileNotFoundError:
            print(f"    ⚠ readelf/objdump not available, skipping dependency analysis")
            break
        except Exception as e:
            print(f"    ⚠ Error analyzing {so_file}: {e}")

    return nvidia_deps

def analyze_libtorch_libs(lib_dir, libs_to_check=None):
    """Analyze the libtorch lib directory to find NVIDIA library dependencies.

    This function checks Linux .so files to determine which NVIDIA libraries
    are bundled vs. which need to be downloaded separately.

    Args:
        lib_dir: Path to libtorch lib directory
        libs_to_check: List of library names to check (e.g., ['cusparse', 'cufft', 'curand'])
                      If None, uses default list from Setup.hs

    Returns:
        Dict mapping library name to needed version, e.g., {'cusparse': '12', 'cufft': '11'}
    """
    if libs_to_check is None:
        # Default libraries from Setup.hs
        libs_to_check = ['cusparse', 'cufft', 'curand']

    print(f"\nAnalyzing Linux libtorch libraries in: {lib_dir}")
    print(f"Libraries to check: {libs_to_check}")

    # Step 1: Extract actual SONAME dependencies
    nvidia_deps = extract_soname_dependencies(lib_dir)

    # Step 2: Scan for bundled .so files
    print("\n2. Scanning for bundled NVIDIA libraries:")

    nvidia_libs_bundled = {lib: False for lib in libs_to_check}

    for filename in os.listdir(lib_dir):
        if not (filename.endswith('.so') or '.so.' in filename):
            continue

        # Check for NVIDIA libraries (full versions, not Lite)
        # Note: cublasLt and cusparseLt are "Lite" versions, not full libraries
        for lib_name in libs_to_check:
            lib_prefix = f"lib{lib_name}"

            # Match patterns like:
            # - libcusparse.so
            # - libcusparse.so.12
            # - libcusparse-abc123.so.12
            if filename.startswith(f"{lib_prefix}.so") or filename.startswith(f"{lib_prefix}-"):
                # Exclude "Lite" versions (e.g., cublasLt, cusparseLt)
                if 'Lt' in filename and lib_name in ['cublas', 'cusparse']:
                    continue

                nvidia_libs_bundled[lib_name] = True
                print(f"  ✓ {lib_name}: {filename}")
                break  # Found this library, move to next file

    # Step 2: Determine what's needed but not bundled
    print("\n2. Determining missing libraries:")

    nvidia_libs_needed = set()

    for lib_name in libs_to_check:
        if not nvidia_libs_bundled.get(lib_name, False):
            nvidia_libs_needed.add(lib_name)
            print(f"  ⚠ {lib_name}: NOT bundled (needs to be downloaded)")
        else:
            print(f"  ✓ {lib_name}: Already bundled")

    print(f"\n3. Summary:")
    bundled_list = [k for k, v in nvidia_libs_bundled.items() if v]
    print(f"  Bundled: {bundled_list if bundled_list else 'None'}")
    print(f"  Need to download: {sorted(nvidia_libs_needed) if nvidia_libs_needed else 'None'}")

    return nvidia_libs_needed

def map_cuda_to_pypi_suffix(cuda_flavor):
    """Map libtorch CUDA flavor (e.g., cu130) to PyPI package suffix (e.g., cu13)."""
    if cuda_flavor.startswith("cu13"):
        return "cu13"
    elif cuda_flavor.startswith("cu12"):
        return "cu12"
    elif cuda_flavor.startswith("cu11"):
        return "cu11"
    else:
        return "cu12"

def get_pypi_wheel_url(package_name, use_nvidia_index=False):
    """Fetch PyPI JSON and find manylinux x86_64 wheel URL."""
    if use_nvidia_index:
        json_url = f"https://pypi.nvidia.com/{package_name}/json"
    else:
        json_url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        with urllib.request.urlopen(json_url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Look through releases to find manylinux x86_64 wheels
        urls = data.get('urls', [])
        for entry in urls:
            filename = entry.get('filename', '')
            url = entry.get('url', '')
            if 'manylinux' in filename and 'x86_64' in filename and url.endswith('.whl'):
                return url, data.get('info', {}).get('version', 'unknown')

        return None, None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        print(f"  ✗ HTTP Error {e.code} for {package_name}")
        return None, None
    except Exception as e:
        print(f"  ✗ Error fetching {package_name}: {e}")
        return None, None

def verify_nvidia_packages(cuda_flavor, nvidia_libs_needed):
    """Verify that the needed NVIDIA packages are available on PyPI.

    Args:
        cuda_flavor: CUDA version flavor (e.g., 'cu130', 'cu121')
        nvidia_libs_needed: Set of library names that need to be downloaded
                           (e.g., {'cusparse', 'cufft', 'curand'})

    Returns:
        True if all needed packages are available, False otherwise
    """
    print(f"\nVerifying NVIDIA packages on PyPI:")
    print(f"CUDA flavor: {cuda_flavor}")

    pypi_suffix = map_cuda_to_pypi_suffix(cuda_flavor)
    use_nvidia_index = cuda_flavor.startswith("cu13")

    if use_nvidia_index:
        print(f"Using NVIDIA PyPI index (pypi.nvidia.com) for {pypi_suffix} packages")
    else:
        print(f"Using standard PyPI index (pypi.org) for {pypi_suffix} packages")

    print()

    all_ok = True

    # Check each needed library
    for lib_name in sorted(nvidia_libs_needed):
        # Convert library name to package name (e.g., 'cusparse' -> 'nvidia-cusparse-cu13')
        package_name = f"nvidia-{lib_name}-{pypi_suffix}"

        print(f"Checking {package_name} (needed)...")
        url, version = get_pypi_wheel_url(package_name, use_nvidia_index)

        if url:
            print(f"  ✓ Found wheel (version {version}):")
            print(f"    {url}")
        else:
            print(f"  ✗ No manylinux x86_64 wheel found")
            all_ok = False

        print()

    if not nvidia_libs_needed:
        print("No additional libraries needed - all are bundled with libtorch!")

    return all_ok

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify NVIDIA library availability for libtorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Check default libraries (cusparse, cufft, curand) for CUDA 13.0
  python3 verify-nvidia-wheels.py cu130

  # Check specific libraries
  python3 verify-nvidia-wheels.py cu121 --libs cublas cusolver

  # Check all common NVIDIA libraries
  python3 verify-nvidia-wheels.py cu130 --libs cusparse cufft curand cublas cusolver
        '''
    )

    parser.add_argument('cuda_flavor',
                       help='CUDA flavor (e.g., cu130, cu121, cu118)')
    parser.add_argument('--libs', nargs='+',
                       help='Libraries to check (default: cusparse cufft curand)')

    args = parser.parse_args()

    cuda_flavor = args.cuda_flavor
    libs_to_check = args.libs if args.libs else None  # None means use defaults

    print(f"=" * 70)
    print(f"Verifying NVIDIA library support for CUDA flavor: {cuda_flavor}")
    print(f"libtorch version: {LIBTORCH_VERSION}")
    print(f"Platform: Linux x86_64 (checking Linux build)")
    print(f"=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Download and extract libtorch
        lib_dir = download_and_extract_libtorch(cuda_flavor, tmpdir)
        if not lib_dir:
            sys.exit(1)

        # Step 2: Analyze what NVIDIA libraries are in libtorch
        nvidia_libs = analyze_libtorch_libs(lib_dir, libs_to_check)

        # Step 3: Verify PyPI packages are available
        all_ok = verify_nvidia_packages(cuda_flavor, nvidia_libs)

    print("=" * 70)
    if all_ok:
        print("✓ All required NVIDIA wheels are available!")
        sys.exit(0)
    else:
        print("✗ Some NVIDIA wheels are missing!")
        sys.exit(1)

if __name__ == "__main__":
    main()
