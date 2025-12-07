#!/bin/bash

# Check if GIT_USER environment variable is set
if [ -z "$GIT_USER" ]; then
  echo "Error: GIT_USER environment variable is not set."
  echo "Usage: GIT_USER=<github_username> ./deploy.sh"
  exit 1
fi

echo "Deploying to GitHub Pages as $GIT_USER..."
GIT_USER=$GIT_USER pnpm run deploy
