# Zephyra

## UNAVAILABLE CURRENTLY: Zephyra is undergoing some major changes and will be updated very soon :) 😁<EOS> 

Zephyra is a from-scratch implementation of a large language model (LLM) tailored for question-answering and chat tasks. This project demonstrates the fundamentals of building and training a transformer-based model, encompassing various stages of data processing, unique model architecture, training, and evaluation.

## Project Overview

Zephyra implements a mixed model architecture, combining different approaches to create an efficient and effective language model. The project focuses on:

- Custom tokenization using Byte Pair Encoding
- Implementation of a transformer-based architecture with rotary positional embeddings
- Training on the CoQA (Conversational Question Answering) dataset
- Efficient training techniques including gradient accumulation and mixed-precision training

## Project Structure

The project has several directories but the key directories among them are structured following:

```
ZEPHYRA
├── data/
│   ├── dataset/
│   │   ├── coqa_train.json
│   │   └── coqa_val.json
│   └── raw/
│       └── coqa-train-v1.0.json
├── src/
│   ├── model/
│   │   ├── attention.py
│   │   ├── layers.py
│   │   └── zephyra.py
│   ├── tokenizer/
│   │   ├── specialTokens.py
│   │   └── tokenizer.py
│   └── utils/
│       ├── dataU.py
│       └── trainArgs.py
├── config.py
├── train.py
└── requirements.txt
```

## Getting Started

To get started with Zephyra:

1. Clone the repository:
   ```
   git clone https://github.com/ajayspatil7/zephyra.git
   cd zephyra
   ```

2. Set up your Python environment (recommend using a virtual environment):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare the CoQA dataset:
   - Download the CoQA dataset from [here](https://stanfordnlp.github.io/coqa/)
   - Place the raw data file in `data/raw/`
   - Run the preprocessing script:
     ```
     python src/utils/coqaPreprocessing.py
     ```

5. Start training:
   ```
   python train.py
   ```

6. Monitor training progress using TensorBoard:
   ```
   tensorboard --logdir=./logs
   ```

## Model Architecture

Zephyra uses a transformer-based architecture with the following key components:

- Rotary Positional Embeddings
- Multi-head Self-Attention
- Feed-forward Neural Networks
- Layer Normalization

The model is defined in `src/model/zephyra.py`.

## Training

The training process includes:

- Mixed precision training for efficiency
- Gradient accumulation to simulate larger batch sizes
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting

Training parameters can be adjusted in `config.py`.

## Evaluation

The model is evaluated on a validation set from the CoQA dataset. Metrics include loss and can be extended to include task-specific metrics like F1 score.

## Contributing

Contributions to Zephyra are welcome! Here are some areas you could consider:

- Enhancing the tokenizer implementations in `src/tokenizer/`
- Improving or adding new model architectures in `src/model/`
- Optimizing the training process in `train.py` and `src/utils/trainArgs.py`
- Adding new datasets or tasks
- Improving documentation and adding tests

Please ensure to follow the project's coding standards and submit pull requests for any new features or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Ajay S Patil

Project Link: [https://github.com/ajayspatil7/zephyra](https://github.com/ajayspatil7/zephyra)

## Acknowledgments

- The CoQA dataset creators
- The open-source AI community for their invaluable resources and inspiration
