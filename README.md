# data-extract-utils

A set of utilities for extracting data from web pages and converting them to markdown format.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/schmitech/data-extract-utils.git
cd data-extract-utils
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dockling Crawler

The `dockling-crawler.py` script converts web pages to markdown files:

```bash
python dockling-crawler.py urls_file.json output_directory
```

where:
- `urls_file.json` is a JSON file containing URLs to process in the format:
  ```json
  [
    {"url": "https://example.com", "file_name": "example.md"},
    {"url": "https://another-example.com", "file_name": "another-example.md"}
  ]
  ```
- `output_directory` is the directory where the markdown files will be saved