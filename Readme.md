# Real State AI

An innovative real estate matching application that leverages advanced AI technologies to generate and match real estate listings based on user preferences. This application utilizes tools like Gradio for user interfaces, LangChain for generating listings, and ChromaDB for storing and querying vector embeddings of real estate listings.

## Features

- **Real Estate Listing Generation**: Utilizes a Large Language Model (LLM) to generate detailed real estate listings.
- **Semantic Search**: Employs vector databases (ChromaDB) to perform semantic searches based on user-defined preferences.
- **Interactive User Interface**: Built with Gradio, allowing users to input their preferences and view matched listings.

## Frameworks Used

- **LangChain**: For generating real estate listings using LLMs.
- **Gradio**: To create interactive web interfaces for real-time user interaction.
- **ChromaDB**: A vector database used for storing and querying embeddings of listings.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip for package installation

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-agent.git
   cd llm-agent
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Start the application:
   ```bash
   python app.py  # Or run the Jupyter Notebook if applicable
   ```

3. Open the Gradio interface in your web browser as instructed by the terminal output.

## Example Usage

After starting the application, input your preferences through the Gradio interface. The application will display real estate listings that match your criteria.