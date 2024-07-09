# Zephyra
Zephyra is a from-scratch implementation of a large language model (LLM) tailored for question-and-answer tasks. This project demonstrates the fundamentals of building and training a transformer-based model, encapsulating various stages of data processing, model training, and evaluation. Enjoy the repo ;)

## Project Structure

The project is organized into several key directories:

```

ZEPHYRA

├── data/

├── experiments/

├── logs/

├── src/

└── temp/

```

### Data

The data/ directory contains various data files used in the project:

- cleanData.txt: Likely a processed, cleaned version of the raw data

- dataSet.csv: A dataset in CSV format

- sample.txt and sampleData.csv: Sample data files, possibly for testing or demonstration

- train.csv and train.txt: Training data in both CSV and text formats

### Experiments

The experiments/ directory seems to be used for different experimental setups:

- __pycache__/: Python cache directory

- cpp/: Possibly C++ related experiments or implementations

- pipe/: Likely contains pipeline experiments or configurations

### Logs

The logs/ directory contains log files:

- zephyra.log: Main log file for the project

### Source Code (src)

The src/ directory contains the main source code for the project:

- __pycache__/: Python cache directory

- generator/: Code related to data/text generation

- model/: Contains the main model implementation

- tokenizer/: Implementation of tokenization algorithms

  - bytepairenc.py: Byte Pair Encoding implementation

  - tokeniser.py: Main tokenizer implementation

  - __init__.py: Initialization file for the module

  - utils.py: Utility functions for the model.

  - zephyra.py: Core Zephyra file. Just like the ```main.py```

  - zephyra.log: Log file for tokeniser

### Temporary Files

The temp/ directory is likely used for temporary files generated during processing or experiments.

## Getting Started

To get started with Zephyra:

1. Clone the repository

2. Set up your Python environment (recommend using a virtual environment)

3. Install required dependencies ```pip install -r requirements.txt```

4. Explore the src/ directory to understand the main components

5. Use the data in the data/ directory for training and testing

## Contributing

Contributions to Zephyra are welcome! Here are some areas you could consider:

- Enhancing the tokenizer implementations in src/tokenizer/

- Improving or adding new generation capabilities in src/generator/

- Optimizing the model in src/model/

- Adding new experiments in the experiments/ directory

- Improving data processing scripts for the files in data/

Please ensure to follow the project's coding standards and submit pull requests for any new features or bug fixes.

## License

MIT

## Contact

Ajay S Patil

Project Link: [https://github.com/ajayspatil7/zephyra](https://github.com/ajayspatil7/zephyra)




