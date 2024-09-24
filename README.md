# Langzap Documentation

<p align="center">
  <img src="langzaplogo.png" alt="Langzap Logo" width="300" height="300">
</p>

**Langzap** is a comprehensive Python library designed to facilitate seamless interaction with Language Model APIs. Currently supporting the **Groq** API, Langzap offers a suite of functionalities including summarization, sentiment analysis, entity extraction, and research capabilities. The library is engineered for flexibility and ease of integration, enabling developers to effortlessly incorporate advanced language model features into their Python applications.

---

## Installation

First, ensure you have all required dependencies. You can install the necessary packages via pip:

```bash
pip install groq dotenv duckduckgo_search
```

## Initializing the Langzap Class

The `Langzap` class is initialized with the following parameters:

- **api_key** *(str, optional)*: Your API key for the provider. If not provided, it will attempt to load from the environment variable.
- **provider** *(str, optional)*: The LLM provider. Defaults to `"groq"`.
- **default_model** *(str, optional)*: The default model used for API calls. Defaults to `"llama-3.1-8b-instant"`.

Example:

```python
from langzap import Langzap

# Initialize Langzap
zap = Langzap(api_key="your_groq_api_key")

# Ask a question
response = zap.ask("What is the capital of France?")
print(response)  # Expected: "Paris"
```

---

## Functions

### `ask(prompt, model=None)`

Generates a response for the given prompt using the LLM API.

**Arguments:**

- **prompt** *(str)*: The input prompt for the model.
- **model** *(str, optional)*: The model to use. Defaults to the `default_model`.

**Returns:**

- *(str)*: The generated response.

**Example:**

```python
response = zap.ask("What is the capital of Germany?")
print(response)  # Expected: "Berlin"
```

---

### `ask_structured(instruction, data, output_instruction, model=None)`

Generates a structured response based on the instruction, data, and output format.

**Arguments:**

- **instruction** *(str)*: The instruction for the model.
- **data** *(str)*: The data to be processed.
- **output_instruction** *(str)*: Instruction on how the output should be formatted.
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(str)*: The structured response.

**Example:**

```python
instruction = "Summarize the given data."
data = "Python is a high-level programming language."
output_format = "Provide a summary in bullet points."
response = zap.ask_structured(instruction, data, output_format)
print(response)
```

---

### `summarize(text, max_words=50, model=None)`

Summarizes a given text to a specified word limit.

**Arguments:**

- **text** *(str)*: The text to be summarized.
- **max_words** *(int, optional)*: The word limit for the summary. Defaults to `50`.
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(str)*: The summary.

**Example:**

```python
long_text = "Python is a widely-used programming language known for its ease of use and readability..."
summary = zap.summarize(long_text, max_words=30)
print(summary)
```

---

### `sentiment(text, model=None)`

Analyzes the sentiment of a given text and returns whether it is positive, negative, or neutral.

**Arguments:**

- **text** *(str)*: The text to analyze.
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(dict)*: A dictionary containing `sentiment` and an `explanation`.

**Example:**

```python
sentiment_analysis = zap.sentiment("I love using Python for data science!")
print(sentiment_analysis)
# Expected output: {'sentiment': 'positive', 'explanation': 'The text expresses a positive emotion towards Python.'}
```

---

### `extract(text, entity_types, model=None)`

Extracts specified entity types from the given text.

**Arguments:**

- **text** *(str)*: The text to analyze.
- **entity_types** *(list)*: A list of entities to extract (e.g., `['person', 'organization']`).
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(dict)*: A dictionary where keys are entity types and values are lists of extracted entities.

**Example:**

```python
entities = zap.extract("Elon Musk is the CEO of SpaceX.", ["person", "organization"])
print(entities)
# Expected output: {'person': ['Elon Musk'], 'organization': ['SpaceX']}
```

---

### `process(data, instruction, output_format=None, model=None)`

Processes data based on the instruction and optional output format.

**Arguments:**

- **data** *(str)*: The input data to process.
- **instruction** *(str)*: The processing instruction.
- **output_format** *(str, optional)*: The desired output format.
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(str)*: The processed output.

**Example:**

```python
data = "Temperature data: 30C, 28C, 35C"
instruction = "Convert temperatures to Fahrenheit."
response = zap.process(data, instruction, output_format="list")
print(response)
```

---

### `search(query, num_results=3)`

Performs a search and caches the results using `duckduckgo_search`.

**Arguments:**

- **query** *(str)*: The search query.
- **num_results** *(int, optional)*: Number of results to return. Defaults to `3`.

**Returns:**

- *(list)*: A list of search results.

**Example:**

```python
search_results = zap.search("Python programming")
print(search_results)
```

---

### `summarize_research(search_results, query, model=None)`

Summarizes a given list of search results for the specified query.

**Arguments:**

- **search_results** *(list)*: The search results to summarize.
- **query** *(str)*: The query associated with the search results.
- **model** *(str, optional)*: The model to use.

**Returns:**

- *(str)*: A summary of the search results.

**Example:**

```python
search_results = zap.search("Latest AI advancements", 5)
summary = zap.summarize_research(search_results, "Latest AI advancements")
print(summary)
```

---

### `research(query, num_results=3, model=None, return_raw_data=False)`

Performs research using search and summarization combined.

**Arguments:**

- **query** *(str)*: The research query.
- **num_results** *(int, optional)*: The number of search results. Defaults to `3`.
- **model** *(str, optional)*: The model to use.
- **return_raw_data** *(bool, optional)*: If `True`, returns both summary and raw data.

**Returns:**

- *(str or tuple)*: Either the summary or a tuple containing the summary and raw search data.

**Example:**

```python
summary, raw_data = zap.research("Quantum computing developments", 3, return_raw_data=True)
print("Summary:", summary)
print("Raw data:", raw_data)
```

---

## Nested Function Usage

**Combining Sentiment Analysis with Summarization:**

```python
text = "I love Python, but JavaScript can be frustrating."
summary = zap.summarize(text, max_words=10)
sentiment = zap.sentiment(summary)
print("Summary:", summary)
print("Sentiment:", sentiment)
```

**Using `extract` within `process`:**

```python
text = "Elon Musk founded SpaceX and Tesla."
entities = zap.extract(text, ["person", "organization"])
response = zap.process(entities, "List the companies founded by Elon Musk.")
print(response)
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

---
