from enum import StrEnum
import logging
import sqlite3
import pickle
from typing import BinaryIO, Optional
from datetime import datetime

import dspy
from src.llm.tools import convert_image


class DocType(StrEnum):
    DOCUMENT = "D"
    PHOTO = "P"
    VOICE = "V"


class DBConn:
    def __init__(self) -> None:
        self.db = sqlite3.connect("data.db")
        self.db.execute("pragma journal_mode=wal")

    def setup_db(self) -> None:
        cur = self.db.cursor()
        # user_id is the user's telegram username
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT UNIQUE NOT NULL, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    );""")

        cur.execute("""CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL, -- 'user' or 'llm'
                    content TEXT,
                    imgs BLOB, -- pickle fmt
                    file_id TEXT,
                    doc_type TEXT,
                    media_group_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
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

    def insert_message(
        self,
        sender: str,
        content: Optional[str] = None,
        imgs: Optional[list[BinaryIO]] = None,
        file_id: Optional[str] = None,
        doc_type: Optional[DocType] = None,
    ) -> int:
        # Sender - "llm" or "user"
        serialized_imgs = pickle.dumps([img.read() for img in imgs]) if imgs else None
        sql = """INSERT INTO messages(sender, content, imgs, file_id, doc_type) VALUES(?, ?, ?, ?, ?) RETURNING message_id"""
        cur = self.db.cursor()
        cur.execute(sql, (sender, content, serialized_imgs, file_id, doc_type))
        msg_id = cur.fetchone()
        self.db.commit()
        return msg_id[0]

    def get_message_by_id(
        self, message_id: int
    ) -> tuple[
        Optional[str], Optional[list[dspy.Image]], Optional[str], Optional[DocType]
    ]:
        """Fetch message content and images by message_id.
            file_id and is_document are used to determine if the message is a document or not.
            LLMS ARE NOT SUPPOSED TO USE THE FILE_ID FOR ANY REASON.
        Returns:
            (content: str | None, images: [dspy.Image] | None, file_id: str | None, is_document: bool).
        """
        sql = """SELECT content, imgs, file_id, doc_type FROM messages WHERE message_id = ?"""
        cur = self.db.cursor()
        cur.execute(sql, (message_id,))
        row = cur.fetchone()

        if not row:
            logging.warning(f"No message found with id: {message_id}")
            return (None, None, None, None)

        if row[1]:
            images = [convert_image(img) for img in pickle.loads(row[1])]
            return (row[0], images, row[2], row[3])

        return row

    def insert_reminder(
        self, user_id: str, content: str, remind_at: datetime, msg_id: int
    ):
        """Insert a new reminder into the reminders table."""
        sql = """INSERT INTO reminders(user_id, reminder_text, remind_at, message_id) VALUES(?, ?, ?, ?)"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id, content, remind_at, msg_id))
        self.db.commit()

    def get_all_pending_reminders(self):
        sql = """SELECT reminder_id, user_id, reminder_text, remind_at FROM reminders
                WHERE status = 'pending' AND remind_at <= CURRENT_TIMESTAMP"""
        cur = self.db.cursor()
        cur.execute(sql)
        return cur.fetchall()

    def get_pending_reminders(self, user_id: str):
        """Get all pending reminders for a specific user.
        Returns:
            List of tuples containing (reminder_id, reminder_text, remind_at)
        """
        sql = """SELECT reminder_id, reminder_text, remind_at FROM reminders
                WHERE user_id = ? AND status = 'pending' AND remind_at <= CURRENT_TIMESTAMP"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id,))
        return cur.fetchall()

    def update_reminder_status(self, reminder_id: int, status: str):
        sql = """UPDATE reminders SET status = ? WHERE reminder_id = ?"""
        cur = self.db.cursor()
        cur.execute(sql, (status, reminder_id))
        self.db.commit()

    def insert_info(self, content: str, msg_id: int, user_id: str) -> int:
        sql = """INSERT INTO information(content, message_id, user_id) VALUES(?, ?, ?) RETURNING info_id"""
        cur = self.db.cursor()
        cur.execute(sql, (content, msg_id, user_id))
        info_id = cur.fetchone()[0]
        self.db.commit()
        return info_id

    def get_info_with_user(self, user_id: str):
        sql = """SELECT content FROM information WHERE user_id = ?"""
        cur = self.db.cursor()
        cur.execute(sql, (user_id,))
        return cur.fetchall()

    def close(self):
        self.db.close()


if __name__ == "__main__":
    dbm = DBConn()
    dbm.setup_db()
