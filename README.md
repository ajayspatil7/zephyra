# Zephyra

Zephyra is a from-scratch implementation of a large language model (LLM) tailored for question-and-answer tasks. This project demonstrates the fundamentals of building and training a transformer-based model, encapsulating various stages of data processing, model training, and evaluation.

## Directory and File Descriptions

- **`__pycache__/`**: Stores Python bytecode files to improve loading times.
  
- **`data/`**: Contains datasets and text files used for training and testing the model.
  - `cleanData.txt`: Processed text data.
  - `dataSet.csv`, `sampleData.csv`, `train.csv`: CSV files containing datasets for various purposes.
  - `sample.txt`, `train.txt`: Sample and training data in text format.

- **`experiments/`**: Holds experimental files and Jupyter notebooks.
  - `hello.asm`: A sample assembly file.
  - `output.o`: Compiled object file.
  - `transformer_train.ipynb`: Jupyter notebook for training and experimenting with the transformer model.

- **`logs/`**: Logging directory.
  - `zephyra.log`: Log file capturing the process and results during execution.

- **`src/`**: Source code directory containing the core components of the Zephyra project.
  - `__init__.py`: Initializes the `src` module.
  - `attention.py`: Implements attention mechanisms.
  - `dataset.py`: Handles dataset loading and manipulation.
  - `dataSetBuilder.py`: Tools for constructing and preprocessing datasets.
  - `encoding.py`: Manages text encoding functions.
  - `tokeniser.py`: Tokenization utilities.
  - `trainingSamplesBuilder.py`: Builds training samples from the raw dataset.
  - `transformerblock.py`: Defines the transformer block architecture.
  - `transformers.py`: High-level transformer model implementations.
  - `utils.py`: Utility functions for various tasks.
  - `zephyra.py`: Main script to run the Zephyra model.

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`README.md`**: Documentation file you are reading right now.
- **`ZephyraTechnicals.pptx`**: PowerPoint presentation explaining technical aspects of the project.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.11 or higher
- Pip (Python package installer)
- Virtualenv (recommended)

### Installation

1. **Clone the repository:**
   ```
   bash
   git clone https://github.com/ajayspatil7/zephyra.git
   cd zephyra 
   
   ```

2. **Creating and Activating venv**
    ```
    python -m venv venv
    source venv/bin/activate  
    # On Windows use `venv\Scripts\activate`             
    ```

2. **Installing Packages**
```
pip install -r requirements.txt
```

### Running the program
<p><i>As of now the model is not fully functional, lots of training is needed. You can run the program without any errors as of now. You can contribute by training the model on as much as data possible</i></p>

Running the program
```bash
python3 src/zephyra.py
```

### Contributing

I welcome contributions to enhance the Zephyra project. Please follow these steps:

1. Fork the repository.
2. Create a new feature branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request(PR).



