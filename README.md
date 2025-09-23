# Telegram Bot Project

This project is a Telegram bot that processes both voice and text messages from users. It utilizes an SQLite database to store user information and message logs. The bot is built using the `aiogram` framework for asynchronous handling of messages.

## Project Structure

```
telegram-bot-project
├── src
│   ├── bot.py               # Entry point of the Telegram bot
│   ├── handlers
│   │   ├── text_handler.py   # Handles incoming text messages
│   │   └── voice_handler.py  # Handles incoming voice messages
│   ├── db
│   │   ├── database.py       # Manages SQLite database connection
│   │   └── models.py         # Defines database models
│   └── config.py             # Configuration settings for the bot
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── .env                      # Environment variables
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd telegram-bot-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure the bot:**
   - Create a `.env` file in the root directory and add your bot token and database URL:
     ```
     BOT_TOKEN=your_bot_token
     DATABASE_URL=sqlite:///your_database.db
     ```

5. **Run the bot:**
   ```
   python src/bot.py
   ```

## Usage

- Send text messages to the bot to receive responses processed by the `TextHandler`.
- Send voice messages to the bot to receive responses processed by the `VoiceHandler`.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.