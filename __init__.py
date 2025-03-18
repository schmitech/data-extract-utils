import re
import os
import asyncio
import json
from pathlib import Path
from tenacity import (
    retry,
    wait_random_exponential,
)
from aiolimiter import AsyncLimiter
from contextlib import asynccontextmanager
from .markdown import load_markdown_files_from_directory, split_markdown
from .token_counting import count_tokens_text, count_tokens_messages, get_available_tokens, are_tokens_available_for_both_conversations
from .prompts import create_answering_conversation_messages, create_extraction_conversation_messages
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Get API key(s) from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_GENAI_MODEL = os.getenv('GOOGLE_GENAI_MODEL', 'gemini-2.0-flash')

if not GOOGLE_API_KEY:
    raise ValueError("No Google API key found. Please set GOOGLE_API_KEY in your .env file.")

# Import Google Generative AI
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"Google Gemini API configured with model: {GOOGLE_GENAI_MODEL}")
except ImportError:
    raise ImportError("Google Generative AI library not found. Please install it with: pip install google-generativeai")

# Rate limiting configuration from environment variables
MODEL_RATE_LIMITS = int(os.getenv('MODEL_RATE_LIMITS', '2000'))
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', int(MODEL_RATE_LIMITS * 0.75)))
MAX_QA_PAIRS = int(os.getenv('MAX_QA_PAIRS', '300'))

# Ensure we do not run too many concurrent requests
throttler = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
api_key_lock = asyncio.Lock()  # Keep lock for potential future multi-key support

def flatten_nested_lists(nested_lists):
    """
    Takes a list of lists as input and returns a flattened list containing all elements.
    
    Args:
        nested_lists (list of lists): A list containing one or more sublists.

    Returns:
        list: A flattened list containing all elements from the input nested lists.
    """
    flattened_list = []

    # Iterate through the nested lists and add each element to the flattened_list
    for sublist in nested_lists:
        flattened_list.extend(sublist)

    return flattened_list

@retry(
    wait=wait_random_exponential(min=15, max=40),
)
async def run_model(messages):
    """
    Asynchronously runs the Google Gemini model on the given messages.
    
    Args:
        messages (list): A list of input messages to be processed by the model.

    Returns:
        str: The model-generated output text after processing the input messages.
    """
    async with api_key_lock:
        # Using a single API key for now, but keeping the lock for future extension
        pass

    # Convert LangChain-style messages to Google Gemini format
    # Typically messages are in format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    
    # Combine all messages into a single prompt
    combined_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            combined_prompt += f"Instructions: {content}\n\n"
        elif role == "user":
            combined_prompt += f"User: {content}\n\n"
        elif role == "assistant":
            combined_prompt += f"Assistant: {content}\n\n"
        else:
            combined_prompt += f"{content}\n\n"

    try:
        # Use a semaphore to limit the number of simultaneous calls
        async with throttler:
            # Create a GenerativeModel instance with the specified model name
            model = genai.GenerativeModel(GOOGLE_GENAI_MODEL)
            
            # Generate content with the combined prompt
            response = await asyncio.to_thread(
                model.generate_content, 
                combined_prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                }
            )
            
            # Extract and return the generated text
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return str(response).strip()
                
    except Exception as e:
        print(f"ERROR ({e}): Could not generate text for an input.")
        return 'ERROR'

def extract_questions_from_output(output):
    """
    Takes a numbered list of questions as a string and returns them as a list of strings.
    The input might have prefixes/suffixes that are not questions or incomplete questions.

    Args:
        output (str): A string containing a numbered list of questions.

    Returns:
        list of str: A list of extracted questions as strings.
    """
    # Define a regex pattern to match questions (lines starting with a number followed by a dot and a space)
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

    # Find all the questions matching the pattern in the input text
    questions = question_pattern.findall(output)

    # Check if the last question is incomplete (does not end with punctuation or a parenthesis)
    if (len(questions) > 0) and (not re.search(r"[.!?)]$", questions[-1].strip())):
        print(f"WARNING: Popping incomplete question: '{questions[-1]}'")
        questions.pop()

    return questions


async def extract_questions_from_text(file_path, text):
    """
    Asynchronously extracts questions from the given text.
    
    Args:
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.

    Returns:
        list of tuple: A list of tuples, each containing the file path, text, and extracted question.
    """
    # Ensure the text can be processed by the model
    text = text.strip()
    num_tokens_text = count_tokens_text(text)

    if not are_tokens_available_for_both_conversations(num_tokens_text):
        # Split text and call function recursively
        print(f"WARNING: Splitting '{file_path}' into smaller chunks.")

        # Build tasks for each subsection of the text
        tasks = []
        for sub_title, sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            task = extract_questions_from_text(sub_file_path, sub_text)
            tasks.append(task)

        # Asynchronously run tasks and gather outputs
        tasks_outputs = await asyncio.gather(*tasks)

        # Flatten and return the results
        return flatten_nested_lists(tasks_outputs)
    else:
        # Run the model to extract questions
        messages = create_extraction_conversation_messages(text)
        output = await run_model(messages)
        questions = extract_questions_from_output(output)

        # Associate questions with source information and return as a list of tuples
        outputs = [(file_path, text, question.strip()) for question in questions]
        return outputs


async def generate_answer(question, source):
    """
    Asynchronously generates an answer to a given question using the provided source text.
    
    Args:
        question (str): The question to be answered.
        source (str): The text containing relevant information for answering the question.

    Returns:
        str: The generated answer to the question.
    """
    # Create the input messages for the chat model
    messages = create_answering_conversation_messages(question, source)
    # Asynchronously run the chat model with the input messages
    answer = await run_model(messages)

    return answer

#---------------------------------------------------------------------------------------------
# FILE PROCESSING

async def process_file(file_path, text, progress_counter, verbose=True, max_qa_pairs=None):
    """
    Asynchronously processes a file, extracting questions and generating answers concurrently.
    
    Args:
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.
        progress_counter (dict): A dictionary containing progress information ('nb_files_done' and 'nb_files').
        verbose (bool): If True, print progress information. Default is True.
        max_qa_pairs (int, optional): Maximum number of question-answer pairs to generate.
            Defaults to  from environment.

    Returns:
        list: A list of dictionaries containing source, question, and answer information.
    """
    if max_qa_pairs is None:
        max_qa_pairs = MAX_QA_PAIRS
        
    questions_file_name = f"{file_path}.json"
    if Path(questions_file_name).is_file():
        with open(questions_file_name, 'r') as input_file:
            questions = json.loads(input_file.read())
    else:
        # Extract questions from the text
        questions = await extract_questions_from_text(file_path, text)

        # Limit the number of questions processed
        questions = questions[:max_qa_pairs]

        with open(questions_file_name, 'w') as output_file:
            json.dump(questions, output_file, indent=2)

    results_filename = f"{file_path}.result.json"
    result = []
    if Path(results_filename).is_file():
        with open(results_filename, 'r') as input_file2:
            result = json.loads(input_file2.read())
    else:
        # Build and run answering tasks concurrently
        tasks = []
        for sub_file_path, sub_text, question in questions:
            task = generate_answer(question, sub_text)
            tasks.append(task)

        tasks_outputs = await asyncio.gather(*tasks)

        # Merge results into a list of dictionaries
        for (sub_file_path, sub_text, question), answer in zip(questions, tasks_outputs):
            result.append({'source': sub_file_path, 'question': question, 'answer': answer})

        with open(results_filename, 'w') as output_file:
            json.dump(result, output_file, indent=2)

    # Update progress and display information if verbose is True
    progress_counter['nb_files_done'] += 1  # No race condition as we are single-threaded
    if verbose:
        print(f"{progress_counter['nb_files_done']}/{progress_counter['nb_files']}: File '{file_path}' done!")

    return result


async def process_files(files, verbose=True):
    """
    Asynchronously processes a list of files, extracting questions and generating answers concurrently.
    
    Args:
        files (list): A list of tuples containing file paths and their respective text content.
        verbose (bool): If True, print progress information. Default is True.

    Returns:
        list: A merged list of dictionaries containing source, question, and answer information.
    """
    # Set up progress information for display
    nb_files = len(files)
    progress_counter = {'nb_files': nb_files, 'nb_files_done': 0}
    if verbose: print(f"Starting question extraction on {nb_files} files.")

    # Build and run tasks for each file concurrently
    tasks = []
    for file_path, text in files:
        task = process_file(file_path, text, progress_counter, verbose=verbose)
        tasks.append(task)

    tasks_outputs = await asyncio.gather(*tasks)

    # Merge results from all tasks
    return flatten_nested_lists(tasks_outputs)

#---------------------------------------------------------------------------------------------
# MAIN

def extract_questions_from_directory(input_folder, verbose=True):
    """
    Extracts questions and answers from all markdown files in the input folder.

    Args:
        input_folder (str): A path to a folder containing markdown files.
        verbose (bool): If True, print progress information. Default is True.

    Returns:
        list: A list of dictionaries containing path, source, question, and answer information.
    """
    # Load input files from the folder
    if verbose: print(f"Loading files from '{input_folder}'.")
    files = load_markdown_files_from_directory(input_folder)

    # Run question extraction tasks
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_files(files, verbose=verbose))

    if verbose: print(f"Done, {len(results)} question/answer pairs have been generated!")
    return results
