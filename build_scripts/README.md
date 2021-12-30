# Build Standalone Executable

This folder contains scripts to build standalone executables from the source python files using PyInstaller. [.github/workflows/build_executable.yaml](/.github/workflows/build_executable.yaml) uses these scripts to build standalone executables on GitHub actions.

## How to use locally

These scripts can also run locally with a Bash shell. On Windows, Git Bash can be used. Note that these scripts are meant to be run in the project root. First, you need to setup a Python venv and install all the necessary dependencies. This needs to be done only once. 

```bash
. ./build_scripts/install_dependencies.sh {your-os}
```

Then, you can run the build script to produce an executable for your platform in the dist folder. Make sure that the Python venv is activated before running this script.

```bash
. ./build_scripts/build.sh {your-os}
```

`{your-os}` is one of `macos`, `windows` and `linux`. Due to the way PyInstaller works, you can only build executable for your platform. For example, you cannot build macos and linux executable on windows.