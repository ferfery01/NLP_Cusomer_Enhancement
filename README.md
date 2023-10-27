# Unified Desktop AI Modules

Welcome to the AI modules repository for the Unified Desktop system!

## Overview

The Unified Desktop system is designed to empower the Walgreens Boots Alliance's pharmacy technicians and pharmacists, offering them advanced functionalities to interact with customers through voice and chat channels. This repository focuses on the AI-powered enhancements that aim to automate and improve various aspects of user interactions.

## Key AI Modules

1. **Automatic Speech Recognition (ASR) / Real-Time Transcription:** Converts real-time voice data into textual format, aiding in downstream processing.

2. **Intent Detection:** Understands the underlying purpose of customer interactions, ensuring accurate and timely responses.

3. **Sentiment Analysis:** Gauges the sentiment or mood of the customer from their interactions, enabling personalized user handling.

4. **Speech Emotion Recognition (SER):** Analyses voice data to understand the emotional tone, further refining the response strategy.

5. **Transcription Summary:** Generates concise summaries of interactions, be it calls, chats, or emails.

6. **Keyword Extraction:** Identifies and extracts important keywords from the interactions for analysis and categorization.

7. **Action Recommendation:** Suggests possible next steps or actions to agents and pharmacists based on the context and content of interactions.

8. **Knowledge Discovery and Management:** Facilitates advanced searching capabilities, content categorization, and personalized search profiles in the system's knowledge base.

## Development Setup

### Python Environment Creation

You must use a virtual environment for an isolated installation of this project. We recommended using either [conda](https://docs.conda.io/en/latest/miniconda.html) or [virtualenv](https://pypi.org/project/virtualenv/).

### `conda` environment

Download and setup [minconda](https://docs.conda.io/en/latest/miniconda.html) to get the `conda` tool.
Once available, create and activate the environment for this project as:

```shell script
conda create -y -n ${NAME_OF_THE_PROJECT} python=3.10
conda activate ${NAME_OF_YOUR_PROJECT}
```

When active, you can de-activate the environment with `conda deactivate`.

### `virtualenv` environment

Suggestion: use [`pipx` to install `virtualenv` in your system via](https://virtualenv.pypa.io/en/latest/installation.html#via-pipx): `pipx install virtualenv`.

To create and activate your environment with `virtualenv`, execute:

```shell script
virtualenv --python 3.10 ~/.venv/${NAME_OF_YOUR_PROJECT}
source ~/.venv/${NAME_OF_YOUR_PROJECT}/bin/activate
```

### Installing Dependencies and Project Code

This project uses [pip](https://pypi.org/project/pip/) for dependency management and [setuptools](https://pypi.org/project/setuptools/) for project code management.

#### Main Dependencies

To install the project’s dependencies & code in the active environment, perform:

```shell script
pip install -r requirements.txt && pip install -e .
```

#### Test & Dev Dependencies

To install the testing and development tools in the environment, do:

```shell script
pip install -e ".[dev]"
```

#### Additional Dependencies

This project requires the command-line tool [ffmpeg](https://ffmpeg.org/) for handling audio input. `ffmpeg` is a widely-used software to record, convert, and stream audio and video. You can install it on your system using the following commands based on your operating system:

```shell script
# For Ubuntu or Debian:
sudo apt update && sudo apt install ffmpeg

# For MacOS (using Homebrew):
brew install ffmpeg
```

## Running Tests and Using Dev Tools

### Testing

TODO: Describe how to run the tests

### Dev Tools

This project uses several tool to maintain high-quality code:

- `mypy` for type checking
- `flake8` for linting
- `isort` for module import organization
- `black` for general code formatting
- `pre-commit` for enforcing use of the above tools

The configuration settings for these tools are defined at the root of the `AI-LAB-RXCONNECT` repository.
****

#### `pre-commit` hooks

**NOTE**: Be sure to first install the `pre-commit` hooks defined in the `.pre-commit-config.yaml` file. To install, execute `pre-commit install` from the repository root while environment is active.

**NOTE**: All code in the project _must_ adhere to using these dev tools _before_ being committed.
****
