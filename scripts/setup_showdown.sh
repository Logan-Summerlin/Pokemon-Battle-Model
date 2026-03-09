#!/usr/bin/env bash
# Setup script for local Pokemon Showdown server.
#
# This script:
# 1. Clones the pokemon-showdown repository
# 2. Installs Node.js dependencies
# 3. Configures the server for local bot play
# 4. Verifies the server starts correctly
#
# Prerequisites:
# - Node.js >= 18 (recommended: use nvm)
# - npm
#
# Usage:
#   bash scripts/setup_showdown.sh [install_dir]
#
# The install directory defaults to ~/pokemon-showdown

set -euo pipefail

INSTALL_DIR="${1:-$HOME/pokemon-showdown}"
REPO_URL="https://github.com/smogon/pokemon-showdown.git"
# Pin to a specific commit for reproducibility
# TODO: Update this to a specific stable commit SHA
COMMIT="HEAD"

echo "=== Pokemon Showdown Local Server Setup ==="
echo "Install directory: $INSTALL_DIR"
echo ""

# Check prerequisites
check_prereqs() {
    if ! command -v node &> /dev/null; then
        echo "ERROR: Node.js is required but not installed."
        echo "Install via: https://nodejs.org/ or use nvm"
        exit 1
    fi

    NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo "ERROR: Node.js >= 18 required (found: $(node --version))"
        exit 1
    fi

    if ! command -v npm &> /dev/null; then
        echo "ERROR: npm is required but not installed."
        exit 1
    fi

    echo "Prerequisites OK: Node.js $(node --version), npm $(npm --version)"
}

# Clone or update the repository
clone_repo() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "Repository already exists at $INSTALL_DIR"
        echo "Pulling latest changes..."
        cd "$INSTALL_DIR"
        git fetch origin
        if [ "$COMMIT" != "HEAD" ]; then
            git checkout "$COMMIT"
        else
            git pull origin master
        fi
    else
        echo "Cloning pokemon-showdown..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        if [ "$COMMIT" != "HEAD" ]; then
            git checkout "$COMMIT"
        fi
    fi

    ACTUAL_COMMIT=$(git rev-parse HEAD)
    echo "Server commit: $ACTUAL_COMMIT"
}

# Install dependencies
install_deps() {
    echo "Installing Node.js dependencies..."
    cd "$INSTALL_DIR"
    npm install
    echo "Dependencies installed."
}

# Configure for local bot play
configure_server() {
    echo "Configuring server for local bot play..."
    cd "$INSTALL_DIR"

    # Create config if it doesn't exist
    if [ ! -f config/config.js ]; then
        if [ -f config/config-example.js ]; then
            cp config/config-example.js config/config.js
        fi
    fi

    # The default config works for local play.
    # Key settings:
    # - Port 8000 (default)
    # - No login server needed for local play
    echo "Server configured for local play on port 8000."
}

# Verify server starts
verify_server() {
    echo "Verifying server starts..."
    cd "$INSTALL_DIR"

    # Start server in background
    node pokemon-showdown start --no-security &
    SERVER_PID=$!

    # Wait for server to start
    sleep 3

    # Check if it's running
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server started successfully (PID: $SERVER_PID)"
        # Stop the server
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server verification complete."
    else
        echo "WARNING: Server may have failed to start. Check for errors above."
    fi
}

# Record the commit SHA for reproducibility
record_commit() {
    cd "$INSTALL_DIR"
    ACTUAL_COMMIT=$(git rev-parse HEAD)

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    echo ""
    echo "=== Setup Complete ==="
    echo "Server installed at: $INSTALL_DIR"
    echo "Commit SHA: $ACTUAL_COMMIT"
    echo ""
    echo "To start the server:"
    echo "  cd $INSTALL_DIR && node pokemon-showdown start --no-security"
    echo ""
    echo "Update configs/environment/showdown.yaml with:"
    echo "  commit: $ACTUAL_COMMIT"
    echo "  install_dir: $INSTALL_DIR"
}

# Main
check_prereqs
clone_repo
install_deps
configure_server
verify_server
record_commit
