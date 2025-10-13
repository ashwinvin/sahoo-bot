# Sahoo Bot

Sahoo Bot is a Telegram bot that leverages LLMs to process, store, and retrieve user data, including text, images, documents, and voice messages. It uses an SQLite database for persistent storage and integrates with FastAPI to serve stored documents via HTTP links.

## Features

- **Telegram Bot**: Handles user messages, documents, images, and voice notes using [aiogram](https://github.com/aiogram/aiogram).
- **LLM Integration**: Uses Gemini and DSPy for advanced query understanding, information extraction, and document generation.
- **Document Storage & Retrieval**: Stores user-uploaded documents and allows retrieval via message IDs.
- **Reminders & Scheduling**: Users can set reminders and schedules, which are managed and triggered by the bot.
- **FastAPI Server**: Serves stored documents from the SQLite database via HTTP endpoints.
- **Chroma Embedding Store**: Stores and retrieves message embeddings for semantic search.


## Setup Instructions

   ```sh
   git clone <repository-url> # Clone the directory
   cd sahoo-bot
   python -m venv venv # Setup virtual env
   source venv/bin/activate 
   pip install -r requirements.txt # install dependencies
   ```

   The API Tokens and other parameters must be set in `.env`, `.env_template` contains the variables to be set. 

   ```sh
   python src/db.py # Initialize the database
   ```

   ```sh
   python src/main.py # Start the bot
   ```

## Usage
Interact with the bot to:

*   **Process & Store Content:** Send messages, images, documents, and voice notes for summarization or secure storage.
*   **Access Stored Data:** Easily retrieve past uploads for clarification, report generation, or to access original files.
*   **Manage Reminders:** Set and receive timely reminders for your important events and tasks.

## License

MIT License