# AI Job Matcher

AI Job Matcher is a Streamlit-based application that uses advanced NLP techniques to match CVs with job postings. The application leverages OpenAI's embeddings and cross-encoder models to provide relevant job recommendations based on CV content.

## Features

- PDF CV upload and processing
- Advanced semantic matching using OpenAI embeddings
- Re-ranking of results using cross-encoder models
- Interactive web interface built with Streamlit
- Comprehensive job metadata display
- Error handling and logging

## Prerequisites

- Python 3.8+
- OpenAI API key
- Chrome database for vector storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sospedra10/ai-job-matcher.git
cd ai-job-matcher
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Required Dependencies

Create a `requirements.txt` file with the following content:

```
streamlit
langchain-openai
langchain-community
chromadb
python-dotenv
torch
transformers
PyPDF2
pandas
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Upload a PDF CV and click "Find Matching Jobs"

## Project Structure

```
ai-job-matcher/
├── app.py                 # Main application file
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
└── chroma_db/            # Vector database storage
```

## Configuration

The application uses a `ModelConfig` class for configuration. You can modify the following parameters:

- `embeddings_model`: OpenAI embeddings model name
- `cross_encoder_model`: Cross-encoder model name
- `chroma_persist_dir`: Directory for vector database storage
- `retriever_k`: Number of initial matches to retrieve
- `rerank_k`: Number of matches to show after re-ranking
- `max_token_length`: Maximum token length for the cross-encoder

## How It Works

1. **CV Processing**: The application accepts PDF CVs and extracts text content.

2. **Initial Matching**: Uses OpenAI embeddings to find potentially relevant job matches from the vector database.

3. **Re-ranking**: Applies a cross-encoder model to re-rank matches based on semantic similarity.

4. **Results Display**: Shows the top matching jobs with detailed metadata and descriptions.

## Error Handling

The application includes comprehensive error handling and logging:
- Input validation for CV uploads
- Model initialization error handling
- Processing pipeline error management
- User-friendly error messages

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Logging

The application uses Python's built-in logging module. Logs are output to the console and include:
- Error messages
- Model initialization status
- Processing pipeline information
- User interaction events

## Security Considerations

- Store API keys securely in the `.env` file
- Don't commit the `.env` file to version control
- Regularly update dependencies for security patches
- Validate and sanitize user inputs

## Limitations

- Currently only supports PDF format for CVs
- Requires active internet connection for OpenAI API
- Limited by OpenAI API rate limits and costs
- Vector database must be pre-populated with job data

## Future Improvements

- Support for additional CV formats (Word, plain text)
- Batch processing capabilities
- Advanced filtering options
- Export functionality for matches
- Integration with job application systems
- Real-time job data updates

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.