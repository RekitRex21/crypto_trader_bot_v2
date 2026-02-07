"""
Core module initialization.
"""
from core.config import config, Config
from core.database import db, DatabaseManager

__all__ = ["config", "Config", "db", "DatabaseManager"]
