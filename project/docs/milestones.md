## Milestones to achieve for Zephyra Resolve

1. Fix the existing model : <thinking>Come up with a new, working & unique different architecture for zephyra which is compitable with zephyra resolve</thinking>. This new architecture will be the base model with basic and common functionalities of a large language model (llm).

2. Training the base model(Zephyra) phase 1 : Train the base zephyra model on a very small and simple dataset for example <example> basic tasks like, simple conversation <thinking> (hello, hi, good morning, etc), simple classification tasks like (watermelon: fruit, potato: vegetable) </thinking> etc.</example> And, save the model for later use.

3. Training the base model(Zephyra) phase 2 : Train the base zephyra model on a <thinking> medium sized dataset such that it has little general knowledge equivalent to GPT2 model or Qwen-0.5 model</thinking>. And, save the model for later use.

4. The single adapters implementation & training : Once our base model is capable enough to do basic tasks after phase 2 training, <thinking> we can start with single adapter trained on a specific domain without having no prior knowledge about this domain. This adapter should know only about the domain we train it on. Based on the prompt given, if the prompt is related to adapter, we send the prompt to that adapter to get the response from that adapter. And will be given to user as the response for the prompt.</thinking>And, save the model for later use.

5. The single adapters testing : Once the single adapter is ready, we test the adapter on which domain it has been trained on by giving prompt, and correct response. And evaluating it with response given by the model. <thinking>we can use different testing methods also to test this model/adapter.</thinking>


6. The multi adapters implementation & training : Once our single adapter is working and giving us expected results we can work on building multiple adapters and each adapters trained on different domains. <thinking>Here can will implement the routing mechanism called 'tiny ada' which will route the user prompt to relevant adapter or We can implement the user prompt classification in the base model which will route the prompt to a specific adapter.</thinking>And, save the model for later use.


7. The multi adapters testing : <thinking>We can use different testing methods to test this model which will send the prompt to a specific adapter based on the prompt and respond back to the user.</thinking> 


8. Concluding the results : Once our architecture is fully ready and tested out, we can conclude the results by writing a research paper on overleaf latex as a IEEE format, On how this technique helps in "Poly domain adaptaion on a unified architecture" as zephyra resolve.