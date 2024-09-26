import groq
import os
from dotenv import load_dotenv
import re
from duckduckgo_search import DDGS
from functools import lru_cache
from pint import UnitRegistry
import sympy

# Load environment variables
load_dotenv()

class Langzap:
    """
    A helper class for interacting with Language Model APIs.

    This class provides methods to initialize the LLM client and send prompts
    to the API, allowing users to easily integrate LLM capabilities into their
    Python programs.

    Currently supports Groq, with plans to expand to other providers in the future.

    Attributes:
        client: The LLM API client.

    Example:
        >>> zap = Langzap()
        >>> response = zap.ask("What is the capital of France?")
        >>> print(response)
    """

    def __init__(self, api_key=None, provider="groq", default_model="llama-3.1-8b-instant"):
        """
        Initialize the Langzap class.

        Args:
            api_key (str, optional): The API key. If not provided, it will be
                                     loaded from the appropriate environment variable.
            provider (str, optional): The LLM provider. Defaults to "groq".
            default_model (str, optional): The default model to use for API calls.
                                           Defaults to "llama-3.1-8b-instant".

        Raises:
            ValueError: If the API key is not provided and not found in environment variables.
            NotImplementedError: If the specified provider is not supported.
        """
        if provider.lower() == "groq":
            if api_key is None:
                api_key = os.getenv("GROQ_API_KEY")
            
            if api_key is None:
                raise ValueError("Groq API key not found. Please provide it or set the GROQ_API_KEY environment variable.")

            self.client = groq.Groq(api_key=api_key)
        else:
            raise NotImplementedError(f"Provider '{provider}' is not currently supported.")

        self.provider = provider
        self.default_model = default_model

    def ask(self, prompt, model=None):
        """
        Get a response from the LLM API for a given prompt.

        Args:
            prompt (str): The input prompt for the model.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            str: The generated response from the LLM API.

        Raises:
            Exception: If an error occurs during the API call.
        """
        model = model or self.default_model
        try:
            if self.provider.lower() == "groq":
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.choices[0].message.content
            else:
                raise NotImplementedError(f"Ask method not implemented for provider '{self.provider}'.")
        except Exception as e:
            raise Exception(f"An error occurred while calling the LLM API: {e}")

    def ask_structured(self, instruction, data, output_instruction, model=None):
        """
        Get a structured response from the LLM API based on an instruction, data, and output format instruction.

        Args:
            instruction (str): The main instruction for the model.
            data (str): The data to be processed.
            output_instruction (str): A natural language instruction on how to structure the output.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            str: The structured response from the LLM API.

        Raises:
            Exception: If an error occurs during the API call.
        """
        prompt = f"""
        Instruction: {instruction}
        Data: {data}
        
        Output Format Instruction: {output_instruction}

        Please provide your response following the output format instruction above.
        """

        return self.ask(prompt, model)

    def summarize(self, text, max_words=50, model=None):
        """
        Summarize the given text.

        Args:
            text (str): The text to summarize.
            max_words (int, optional): Maximum number of words for the summary. Defaults to 50.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            str: A concise summary of the input text.
        """
        instruction = f"Summarize the following text in no more than {max_words} words:"
        output_instruction = f"Provide a concise summary of the text, using at most {max_words} words."
        return self.ask_structured(instruction, text, output_instruction, model)

    def sentiment(self, text, model=None):
        """
        Analyze the sentiment of the given text.

        Args:
            text (str): The text to analyze.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            dict: A dictionary containing the sentiment (positive, negative, or neutral) and a brief explanation.
        """
        instruction = "Analyze the sentiment of the following text:"
        output_instruction = "Provide the sentiment as either 'positive', 'negative', or 'neutral', followed by a brief explanation. Format the response as a Python dictionary with keys 'sentiment' and 'explanation'."
        response = self.ask_structured(instruction, text, output_instruction, model)
        
        # Clean the response and extract the dictionary
        response = response.strip()
        if response.startswith("```python"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return eval(response.strip())  # Convert the cleaned string response to a dictionary
        except:
            # If evaluation fails, return a default dictionary
            return {"sentiment": "unknown", "explanation": "Failed to parse the sentiment analysis result."}

    def extract(self, text, entity_types, model=None):
        """
        Extract specified entity types from the given text.

        Args:
            text (str): The text to extract entities from.
            entity_types (list): A list of entity types to extract (e.g., ['person', 'organization', 'location']).
            model (str, optional): The model to use for the API call. If not provided, uses the default model.

        Returns:
            dict: A dictionary where keys are entity types and values are lists of extracted entities.
        """
        instruction = f"Extract the following entity types from the text: {', '.join(entity_types)}"
        output_instruction = f"Provide the extracted entities as a Python dictionary where keys are entity types and values are lists of extracted entities. Only include the dictionary in your response."
        response = self.ask_structured(instruction, text, output_instruction, model)
        
        # Clean the response
        response = response.strip()
        if response.startswith("```python"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            # Safely evaluate the string as a Python expression
            import ast
            return ast.literal_eval(response.strip())
        except Exception as e:
            print(f"Error parsing extracted entities: {e}")
            return {entity_type: [] for entity_type in entity_types}

    def process(self, data, instruction, output_format=None, model=None):
        """
        A general-purpose function to process data based on given instructions and output format.

        Args:
            data (str): The input data to be processed.
            instruction (str): The instruction for processing the data.
            output_format (str, optional): Specification for the desired output format.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            str: The processed result from the LLM API.

        Raises:
            Exception: If an error occurs during the API call.
        """
        prompt = f"""
        Instruction: {instruction}

        Data: {data}

        {f"Output Format: {output_format}" if output_format else ""}

        Please process the data according to the instruction{" and provide the output in the specified format" if output_format else ""}.
        """

        return self.ask(prompt, model)

    @lru_cache(maxsize=100)
    def search(self, query, num_results=3):
        """Perform a search and cache the results."""
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num_results))

    def summarize_research(self, search_results, query, model=None):
        """Summarize the given search results."""
        instruction = f"Summarize the following search results for the query: '{query}'"
        data = str(search_results)
        return self.ask_structured(instruction, data, "Provide a concise summary of the research findings.", model)

    def research(self, query, num_results=3, model=None, return_raw_data=False):
        """
        Research a query using search and summarization.

        Args:
            query (str): The research query.
            num_results (int, optional): Number of search results to fetch. Defaults to 3.
            model (str, optional): The model to use for processing.
            return_raw_data (bool, optional): If True, return both the summary and raw search results.

        Returns:
            str or tuple: A summary of the research findings, or a tuple containing the summary and raw search results.
        """
        raw_results = self.search(query, num_results)
        summary = self.summarize_research(raw_results, query, model)
        
        return (summary, raw_results) if return_raw_data else summary

    def translate(self, text, target_language, model=None):
        """
        Translate the given text to the specified target language.

        Args:
            text (str): The text to translate.
            target_language (str): The language to translate the text into.
            model (str, optional): The model to use for the API call. 
                                   If not provided, uses the default model.

        Returns:
            str: The translated text.
        """
        instruction = f"Translate the following text to {target_language}:"
        output_instruction = f"Provide the translation in {target_language}."
        return self.ask_structured(instruction, text, output_instruction, model)

    def convert_units(self, query, model=None):
        """
        Detect and convert units based on the given query.

        Args:
            query (str): The query containing unit conversion request.
            model (str, optional): The model to use for the API call.
                                   If not provided, uses the default model.

        Returns:
            str: The result of the unit conversion.
        """
        ureg = UnitRegistry()

        instruction = "Analyze the following query and extract the value, source unit, and target unit for conversion:"
        output_instruction = "Provide the extracted information as a Python dictionary with keys 'value', 'from_unit', and 'to_unit'. Only include the dictionary in your response."
        
        extraction = self.ask_structured(instruction, query, output_instruction, model)
        
        try:
            # Safely evaluate the string as a Python expression
            import ast
            extracted_info = ast.literal_eval(extraction.strip())
            
            value = float(extracted_info['value'])
            from_unit = ureg(extracted_info['from_unit'])
            to_unit = ureg(extracted_info['to_unit'])
            
            result = value * from_unit.to(to_unit)
            
            preparsed = f"{value} {from_unit} is equal to {result:.4f} {to_unit}"
            parsed = self.process(preparsed, "Remove extras from output", "Provide the parsed and converted result as a string. Only include the string in your response.")
            return parsed
        except Exception as e:
            return f"Error in unit conversion: {e}"

    def calculate(self, query, model=None):
        """
        Evaluate a mathematical expression or text-based math query using sympy.

        Args:
            query (str): The mathematical expression or text-based math query to evaluate.
            model (str, optional): The model to use for the API call.
                                   If not provided, uses the default model.

        Returns:
            str: The result of the calculation.
        """
        instruction = "Convert the following text or mathematical expression into a valid Python mathematical expression:"
        output_instruction = "Provide only the Python mathematical expression as a string, without any additional text or explanation."
        
        prepared_expression = self.ask_structured(instruction, query, output_instruction, model)
        
        try:
            result = sympy.sympify(prepared_expression).evalf()
            raw_output = f"{query} = {result}"
            parsed_output = self.process(raw_output, "Correct any grammar issues in this mathematical statement", "Provide the grammatically correct statement as a string. Only include the corrected string in your response.")
            return parsed_output
        except Exception as e:
            return f"Error in calculation: {e}"

# Usage example:
if __name__ == "__main__":
    # Initialize with a custom default model
    zap = Langzap(default_model="llama-3.1-8b-instant")
    
    # This will use the default model (llama-3.1-70b-chat)
    response = zap.ask("What is the capital of France?")
    print("Response using default model:", response)

    # This will use a specific model, overriding the default
    response = zap.ask("What is the capital of Germany?", model="llama-3.1-8b-instant")
    print("Response using specific model:", response)

    # Other examples remain the same, but will use the default model unless specified otherwise

    # Example of using the research function
    research_result = zap.research("Latest advancements in quantum computing", model="llama-3.1-8b-instant")
    print("Research result:", research_result)

    # Example of a more complex calculation
    complex_calc = zap.calculate("integrate(x^2 + 2*x + 1, x)")
    print("Complex calculation result:", complex_calc)

    #example of a simple cacluation
    simple_calc = zap.calculate("1+2")
    print("Simple calculation result:", simple_calc)

    
