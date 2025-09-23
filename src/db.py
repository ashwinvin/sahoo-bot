import sqlite3
import pickle
from typing import BinaryIO, Optional
from datetime import datetime


class DBConn:
    def __init__(self) -> None:
        self.db = sqlite3.connect("data.db")
        self.db.execute("pragma journal_mode=wal")

    def setup_db(self) -> None:
        cur = self.db.cursor()
        # user_id is the user's telegram username
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT UNIQUE NOT NULL, 
                        notion_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );""")

        cur.execute("""CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id));
                """)

        cur.execute("""CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    sender TEXT NOT NULL, -- 'user' or 'llm'
                    content TEXT,
                    imgs BLOB, -- pickle fmt
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id));
                    """)

        cur.execute("""CREATE TABLE IF NOT EXISTS information(
                    info_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    content TEXT NOT NULL, 
                    message_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users (user_id),
                    FOREIGN KEY(message_id) REFERENCES messages(message_id));""")

        # status: 'pending', 'triggered', 'completed', 'dismissed'
        cur.execute("""CREATE TABLE IF NOT EXISTS reminders(
                    reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message_id INTEGER,
                    reminder_text TEXT NOT NULL,
                    remind_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending' NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id));
                    """)

        self.db.commit()
        self.db.close()

    def insert_user(self, username: str):
        sql = """INSERT OR IGNORE INTO users(user_id) VALUES(?)"""
        cur = self.db.cursor()
        cur.execute(sql, (username,))
        self.db.commit()

    def insert_conversation(self, topic: str, user_id: str) -> int:
        sql = """INSERT INTO conversations(topic, user_id) VALUES (?, ?) RETURNING conversation_id"""
        cur = self.db.cursor()
        cur.execute(sql, (topic, user_id))
        convo_id = cur.fetchone()
        self.db.commit()
        return convo_id[0]

    def get_user_conversations(self, user_id: str):
        sql = """SELECT conversation_id, topic FROM conversations WHERE user_id = ?"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id,))
        return cur.fetchall()

    def insert_message(
        self,
        convo_id: int,
        sender: str,
        content: Optional[str] = None,
        imgs: Optional[list[BinaryIO]] = None,
    ) -> int:
        # Sender - "llm" or "user"
        serialized_imgs = pickle.dumps([img.read() for img in imgs]) if imgs else None
        sql = """INSERT INTO messages(conversation_id, sender, content, imgs) VALUES(?, ?, ?, ?) RETURNING message_id"""
        cur = self.db.cursor()
        cur.execute(sql, (convo_id, sender, content, serialized_imgs))
        msg_id = cur.fetchone()
        self.db.commit()
        return msg_id[0]

    def insert_reminder(
        self, user_id: str, content: str, remind_at: datetime, msg_id: int
    ):
        sql = """INSERT INTO reminders(user_id, reminder_text, remind_at, message_id) VALUES(?, ?, ?, ?)"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id, content, remind_at, msg_id))
        self.db.commit()

    def get_pending_reminders(self):
        sql = """SELECT reminder_id, user_id, reminder_text, remind_at FROM reminders WHERE status = 'pending' AND remind_at <= CURRENT_TIMESTAMP"""
        cur = self.db.cursor()
        cur.execute(sql)
        return cur.fetchall()

    def insert_info(self, content: str, msg_id: int, user_id: str):
        sql = (
            """INSERT INTO information(content, message_id, user_id) VALUES(?, ?, ?)"""
        )
        cur = self.db.cursor()
        cur.execute(sql, (content, msg_id, user_id))
        self.db.commit()

    def get_info_with_user(self, user_id: str):
        sql = """SELECT content FROM information WHERE user_id = ?"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id,))
        return cur.fetchone()

    def close(self):
        self.db.close()


if __name__ == "__main__":
    dbm = DBConn()
    dbm.setup_db()
