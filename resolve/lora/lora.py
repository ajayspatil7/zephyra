"""
------------------------------Basics------------------------------
Experiment name : Poly-Domain-Unified-Architecture
Author          : Me
Area            : Large Language Models, Deep Neural Networks, Transformers

Abstract        : Embedding multiple different adapters trained on different different domains into one unified architecture.
Motivation      : Reduced computation, Increased efficiency, Multiple domain adaptation, One architecture.
Benifits        : Modular architecture, Computational efficiency, Adapters specialisation, Scalability into different domains
Considerations  : Domain Routing mechanism -"tiny ada", Multi domain queries, Training complexity, Model size, Inference latency drop due to "tiny ada"
------------------------------Implementation-----------------------
Possible implementation strategy
    
    1) Theoritical foundation
        a) Literature review on current adapters methods and their tradeoffs for domain adaptations.
        b) Comprehensive summary of these state-of-the-art methods.
    
    2) Architecture design
        a) Outlining high-level architecture.
        b) Defining the model components.
    
    3) Base implementation
        a) Choose a base model and implement using frameworks.
        b) Get the results and summarise the reports.
    
    4) Adapter implementation
        a) Implement the LoRA embeddings from scratch.
        b) Test it with single domain and verify the results.
    
    5) Multi adapter architecture
        a) Implement multiple adapters for the same base model & ensure they work.
        b) Develop a mechanism to switch between the adapters.
    
    6) Routing mechanism implementation
        a) Desigining the routing mechanism.
        b) Starting with simple classifying approach for routing.
    
    7) Data collection and preprocessing
        a) Collecting domain specific data.
        b) Preparing the data for training.
    
    8) Designing the training pipeline.
        a) A pipeline for training individual adapters.
        b) Implementing a mechanism to train routing adapter.
    
    9) Evaluation and Prototype development.
        a) Evaluating individual adapters, Evaluating "tiny ada" on classification, Evaluating overall system performance
        b) Integrating all components in and starting a small scale protptype.       =
"""
