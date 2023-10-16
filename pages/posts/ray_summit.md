---
title: "What I learned attending Large Language Model track talks at the Ray Summit 2023"
date: 2023/10/04
description: Reflecting on the topics discussed as part of the LLM track in the Ray Summit 2023
tag: llm, nlp, ray, summit
author: Marwan
---

### Anyscale - Robert Nishihara
#### Anyscale endpoints

As of the time of writing (Oct 4, 2023), there are only a few companies that offer LLMs as a service - i.e. they offer API endpoints that you can call to either perform inference on a model or to fine-tune a model. The most prominent ones are:

The most prominent ones are:
- [OpenAI API](https://openai.com/blog/openai-api)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [Anyscale Endpoints](https://app.endpoints.anyscale.com/landing)

Interesting enough - anyscale's API pricing is the lowest among the three ! Anyscale prouds to have a price of $1/million tokens for llama-70b-chat, while OpenAI's API's gpt-3.5-turbo is $1.5/million tokens and Amazon's Bedrock Anthropic claude is $1.102/million tokens. 

While it is not really an apples-to-apples comparison, it is still interesting to see that Anyscale is able to offer a very competitive pricing - which goes to prove that using Ray as a backend for LLMs is a very cost-effective solution.


### Anyscale - Sofian Hnaide
#### Anyscale doctor - a new tool for debugging and monitoring Ray applications

Sofian did a really cool demo of anyscale doctor which is essentially an agent that can perform automated diagnosis and remediation of common issues that developers face.

He started by loading up a jupyter notebook where the first cell is an import statement that attempts to load a module but fails with a ModuleNotFoundError. The doctor tool is able to properly diagnose the issue and suggest a fix by installing the missing module. 

The second example he showed the anyscale doctor fixing a gnarly issue where the user was trying to execute ray code. Whether the agent's chain of reasoning was hardcoded or learned is not clear, but it seemed like a promising prototype.


### Anyscale - Goku Mohandas and Philipp Moritz
#### Developing and Serving RAG-Based LLM Applications in Production

In this session, Goku and Philipp go over their extensive blog post on [Developing and Serving RAG-Based LLM Applications in Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1). 

The blog post takes a deep-dive into an attempt at build a retrieval-augmented generation (RAG) model which is deployed as a service on the [ray documentation website](https://docs.ray.io/en/latest/) which you can access by clicking the "Ask AI" button in the lower right corner of the page.

Here are some key takeaways from the talk:
- A good starting point when approaching any problem is to try the LLM on its own. Avoid premature optimization by trying to build a complex system from the get-go.
- There are many components that need to be optimally configured to get the best performance out of adding a retrieval component to a generation model. These include:
    - Parsing of the documents, as an example:
        - Handling links in text
        - Handling code blocks in text
        - Handling images in text
        - Handling syntax (e.g. markdown/html) that is used to format the text
    - Creating nodes from the documents that will be embedded and indexed for retrieval and primarily the blog post focused on tuning chunking strategies - e.g.:
        - How to split the text, e.g.
            - by paragraph
            - by section
        - The maximum size of the chunk to use
    - Embedding the nodes:
        - The choice of the embedding model to use (the blog highlights the fact that even though an embedding model might rank high on a benchmark, it might not be the best choice for your application)
    - Indexing the embedded vectors (i.e. a vector database solution):
        - There are new solutions that are being developed that are optimized for LLMs, e.g.
            - [weaviate](https://weaviate.io/)
            - [pinecone](https://www.pinecone.io/)
        - There are old solutions that are being extended to support LLMs, e.g.
            - [pg-vector](https://github.com/pgvector/pgvector) - a postgres extension
            - [elasticsearch](https://www.elastic.co/enterprise-search/vector-search)
    - The number of nodes to retrieve given a query
    - The LLM models to use for generation
- Building a proper evaluation framework is key to understanding the performance of the system and to be able to objectively compare different configurations. The approach to evaluation is to evaluate:
    - the retrieval component on its own
    - the LLM conditioned on proper retrieval
    - the end-to-end system (retrieval + LLM)
- Constructing a proper evaluation dataset is key to being able to evaluate the system. The blog post suggests the following:
    - For the retrieval component
- The blog post suggests producing the following metrics:
    - For the retrieval component:
        - Recall@k
        - Mean Reciprocal Rank (MRR)
    - For the LLM component:
        - Perplexity
        - BLEU
        - ROUGE


### Langchain/Langsmith - Harrison Chase
#### Evaluating and Managing LLM Applications with the Langsmith Project


### LLamaIndex - Jerry Liu
#### Practical data considerations for Building Production-Ready LLM Applications


### Gorilla - Shishir Patil
#### Large Language Models connected with Massive APIs


### Anyscale - Waleed Kadous
#### Open Source LLMs: Viable for production? or a low-quality toy?


### You.com - Bryan McCann
#### NLP and the future of Search


### Meta - Joseph Spisak
#### LLama, Scaling Up LLMs in an Open Ecosystem


### Anyscale training - Adam Breindel
#### Learn how to build and productionize LLM-Powered Applications with Ray and Anyscale


### Anyscale training - Goku Mohandas, Simon Suo, Amog Kamsetty
#### Practical Data and Evaluation Considerations for Building Production-Ready LLM Applications with LLamaIndex and Ray





