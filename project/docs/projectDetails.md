# Poly-Domain-Unified-Architecture

**Author:** Me (Ajay S Patil)  
**Base model name**: Zephyra  
**Research** : Poly domain unified Large language model architecture  
**Adapted model** : Zephyra Resolve  
**Area:** Large Language Models, Deep Neural Networks, Transformers, Linear algebra

## Abstract
The attempt to use adapters based fine tuning on a large language models are giving better results compared traditional fine tuning but, it lacks in adapting well to more than one domain in a single model. So, I have come up a unified architecture which supports more than one domain adapataions using the power of adapters like LoRA, QLoRA, etc. And, a very new mechanism to route between or switch between the different adapters & I call it 'tiny ada'. Tine ada, is a tiny classification model inside a large language model which helps in activating a particular adapter of a particular domain based on the prompt. If there are 5 different adapters each with their own domain adaptation; tiny ada will have 5 different classes to classify based on the prompt.

## Brief Introduction
Fine tuning a large language model on a specific domain though reduces the time complexity and computatinal cost as compared to full model pre-training but, still lacks in adapting to new domain completely and, can have problem adapting to multiple domains in one unified architecture. So, our architecture comprises of different adapters of different domain to attend that particular domain based on the routing mechanism 'tinyAda'

## Motivation
- Reduced computation
- Increased efficiency
- Multiple domain adaptation
- One unified architecture

## Benefits
- Modular architecture
- Computational efficiency
- Adapters specialization
- Scalability into different domains

## Considerations
- Domain Routing mechanism -"tiny ada"
- Multi-domain queries
- Training complexity
- Model size
- Inference latency drop due to "tiny ada"

## Implementation

### Possible implementation strategy

1. **Theoretical foundation**
    - Literature review on current adapters methods and their tradeoffs for domain adaptations.
    - Comprehensive summary of these state-of-the-art methods.

2. **Architecture design**
    - Outlining high-level architecture.
    - Defining the model components.

3. **Base implementation**
    - Choose a base model and implement using frameworks.
    - Get the results and summarize the reports.

4. **Adapter implementation**
    - Implement the LoRA embeddings from scratch.
    - Test it with single domain and verify the results.

5. **Multi adapter architecture**
    - Implement multiple adapters for the same base model & ensure they work.
    - Develop a mechanism to switch between the adapters.

6. **Routing mechanism implementation**
    - Designing the routing mechanism.
    - Starting with a simple classifying approach for routing.

7. **Data collection and preprocessing**
    - Collecting domain-specific data.
    - Preparing the data for training.

8. **Designing the training pipeline**
    - A pipeline for training individual adapters.
    - Implementing a mechanism to train routing adapter.

9. **Evaluation and Prototype development**
    - Evaluating individual adapters, Evaluating "tiny ada" on classification, Evaluating overall system performance.
    - Integrating all components and starting a small scale prototype.
