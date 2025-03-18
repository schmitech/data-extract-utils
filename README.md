# data-extract-utils

A collection of utilities for extracting, processing, and analyzing data from various sources.

## Installation

```bash
# Clone the repository
git clone https://github.com/schmitech/data-extract-utils.git
cd data-extract-utils

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Tools

### 1. Dockling Crawler

Converts web pages to markdown files with configurable options.

```bash
python dockling-crawler.py urls_file.json output_directory
```

**Input Format:**
```json
[
  {"url": "https://example.com", "file_name": "example.md"},
  {"url": "https://another-example.com", "file_name": "another-example.md"}
]
```

### 2. Question Extractor 🧐

Automatically generates question/answer pairs from markdown documents using AI models.

#### Configuration

Create a `.env` file (copy from `.env.example`) with your API credentials:

## Question Extractor 🧐

Large language models can be instruction tuned with a set of questions and answers.
However, to further fine-tune a model *on your own data*, you need a large number of questions and answers about your data.
Producing those questions and answers can be a lot of manual work.

This repository lets you use AI models (Google Gemini or OpenAI) to extract question/answer pairs automatically from existing textual data, eliminating all manual work.

## Installation

To run this code, you will need to clone this repository then install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

### Google Gemini Setup
1. Create a `.env` file (copy from .env.example)
2. Add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_GENAI_MODEL=gemini-2.0-flash
   ```
3. Optionally, you can adjust other parameters:
   ```
   MAX_QA_PAIRS=300
   MAX_CONCURRENT_REQUESTS=5
   ```

## Usage

This script is designed to turn a folder of markdown (`.md`) documents into a `.json` file containing a list of questions, answers and paths to the source documents that were used to produce them.

### Using Google Gemini (Recommended)

```bash
python google_question_extractor.py --input ./docs --output ./questions.json
```

Additional options:
- `--no-cache`: Skip cache and regenerate all questions and answers
- `--quiet` or `-q`: Run with minimal output

## Output Format

The output JSON file contains an array of objects with these fields:
- `source`: Path to the source file
- `question`: Generated question
- `answer`: Generated answer

## Caching

For efficiency, both versions of the script cache intermediate results:

- `{file_path}.json`: Cached questions for the file
- `{file_path}.result.json`: Cached question-answer pairs

This allows the script to run quickly on subsequent executions and resume interrupted processes. Use the `--no-cache` flag with the Google version to bypass caching.

## Inner-workings

The code processes all markdown files concurrently for speed and efficiency:

1. For each file, it extracts 5-10 relevant questions using carefully designed prompts
2. It then generates comprehensive answers to each question based only on the source text
3. The results are combined into a single JSON file

If a text is too long to be processed, it is split along its highest markdown heading level (the process can be repeated recursively if needed until we get down to single paragraphs).

### Google Gemini Version

The Google Gemini implementation uses modern async processing for maximum efficiency and throughput. It includes better error handling, improved prompting, and command-line arguments for flexibility.

Performance will vary based on your API quota and the chosen model, but the Google Gemini implementation is generally faster and more cost-effective than the OpenAI version.

## Potential improvements

- Add more configuration options (temperature, number of questions per file, etc.)
- Add option to use different models for question generation vs. answering
- Implement better text splitting strategies for very large documents
- Add support for file formats beyond markdown
- Create a web interface for easier use
