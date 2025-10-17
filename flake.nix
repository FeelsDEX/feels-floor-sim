{
  description = "Python development environment with Jupyter and polars";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Build SALib from GitHub since PyPI source is not available
        salib = pkgs.python3Packages.buildPythonPackage rec {
          pname = "SALib";
          version = "1.5.1";
          pyproject = true;
          
          src = pkgs.fetchFromGitHub {
            owner = "SALib";
            repo = "SALib";
            rev = "v${version}";
            sha256 = "sha256-aJmX0FzvgWSmp0LQv/V8vNg4aAp3xd4HlGoTk1Ce8Is=";
          };
          
          build-system = with pkgs.python3Packages; [
            hatchling
            hatch-vcs
          ];
          
          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            scipy
            matplotlib
            pandas
            multiprocess
          ];
          
          # Skip tests to avoid test dependencies
          doCheck = false;
          
          meta = with pkgs.lib; {
            description = "Sensitivity Analysis Library in Python";
            homepage = "https://github.com/SALib/SALib";
            license = licenses.mit;
          };
        };
        
        # Build AgentPy from PyPI with environment variables to skip pytest-runner
        agentpy = pkgs.python3Packages.buildPythonPackage rec {
          pname = "agentpy";
          version = "0.1.5";
          format = "setuptools";
          
          src = pkgs.python3Packages.fetchPypi {
            inherit pname version;
            sha256 = "sha256-aQuCQ238thSiDUsRJ4G1JAlXrlbRPz2mOA0hiCr5fRM=";
          };
          
          # Set environment variables to avoid pytest-runner
          preBuild = ''
            export SETUPTOOLS_SCM_PRETEND_VERSION="${version}"
            export SKIP_PYTEST_RUNNER=1
          '';
          
          # Patch setup.cfg to remove pytest-runner requirement
          postPatch = ''
            if [ -f setup.cfg ]; then
              sed -i '/pytest-runner/d' setup.cfg
            fi
            if [ -f setup.py ]; then
              sed -i 's/pytest-runner[^,]*,\?//g' setup.py
              sed -i 's/,\s*pytest-runner[^,]*//g' setup.py
            fi
          '';
          
          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            scipy
            matplotlib
            networkx
            pandas
            joblib
            salib
          ];
          
          # Skip tests and dependency checks for this older package
          doCheck = false;
          dontCheckRuntimeDeps = true;
          
          meta = with pkgs.lib; {
            description = "Agent-based modeling in Python";
            homepage = "https://github.com/jofmi/agentpy";
            license = licenses.bsd3;
          };
        };
        
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          jupyter
          polars
          pyarrow  # Required for polars .to_pandas() conversion
          matplotlib
          numpy
          seaborn
          ipykernel
          pytest
          pytest-cov
          typer
          pandas
          networkx
          scipy
          joblib
          agentpy  # Now include agentpy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            nodejs  # for Jupyter extensions
          ];
          
          shellHook = ''
            # Add python symlink if it doesn't exist
            if ! command -v python >/dev/null 2>&1; then
              mkdir -p $PWD/.nix-bin
              ln -sf ${pythonEnv}/bin/python3 $PWD/.nix-bin/python
              export PATH="$PWD/.nix-bin:$PATH"
            fi
            
            # Setup messages and kernel registration
            echo "Python development environment loaded"
            
            # Auto-register Jupyter kernel if not already registered
            KERNEL_NAME="feels-floor-sim"
            if ! jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
              echo "Registering Jupyter kernel: $KERNEL_NAME"
              python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Feels Floor Sim (Nix)"
            else
              echo "Jupyter kernel '$KERNEL_NAME' already registered"
            fi
            
            # Update VSCode settings with current Python path
            if [ -f ".vscode/settings.json" ]; then
              # Create a temporary settings file with relative/dynamic paths
              cat > .vscode/settings.json.tmp << EOF
{
  "python.defaultInterpreterPath": "./.vscode/python",
  "python.terminal.activateEnvironment": false,
  "jupyter.kernels.filter": [
    {
      "path": "$HOME/Library/Jupyter/kernels/feels-floor-sim/kernel.json",
      "type": "jupyterKernel"
    }
  ],
  "jupyter.defaultKernel": "feels-floor-sim",
  "jupyter.interactiveWindow.creationMode": "perFile",
  "files.associations": {
    "*.ipynb": "jupyter-notebook"
  }
}
EOF
              mv .vscode/settings.json.tmp .vscode/settings.json
              
              # Create a symlink to the current Python interpreter
              ln -sf "$(which python3)" .vscode/python
              echo "Updated .vscode/settings.json with dynamic Python symlink"
            fi
          '';
        };
      });
}