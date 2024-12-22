import json
import os
from datetime import datetime
from typing import Dict, Optional, Any

class SessionManager:
    """Manages session setup and provides session details to other components."""

    def __init__(self, session_id: str, base_dir: str = "history"):
        """Initialize the session manager with a session ID."""
        self.session_id = session_id
        self.base_dir = base_dir

        # Set up session directory structure
        self.session_dir = os.path.join(base_dir, "sessions", session_id)
        self.contexts_dir = os.path.join(base_dir, "contexts", session_id)

        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.contexts_dir, exist_ok=True)

        # Define file paths
        self.session_file = os.path.join(self.session_dir, "session_data.json")
        self.context_history_file = os.path.join(self.contexts_dir, "chat_context_history.json")
        self.current_context_file = os.path.join(self.contexts_dir, "current_context.txt")
        self.full_history_file = os.path.join(base_dir, "full_history.json")
        self.current_session_file = os.path.join(base_dir, "current_session.txt")

        # Initialize session data
        self._initialize_session()
        self._set_as_current()

    def _initialize_session(self) -> None:
        """Initialize basic session data."""
        session_data = {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "benchmarks": {
                "models": {}
            },
            "session_totals": {
                "total_time": 0,
                "total_chunks": 0,
                "total_words": 0
            },
            "interactions": []
        }

        # Write initial session data
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        # Initialize or update full history
        self._update_full_history()

    def _set_as_current(self) -> None:
        """Set this session as the current active session."""
        os.makedirs(os.path.dirname(self.current_session_file), exist_ok=True)
        with open(self.current_session_file, "w") as f:
            f.write(self.session_id)

    def _load_session_data(self) -> Dict:
        """Load current session data."""
        try:
            with open(self.session_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Reinitialize if missing
            self._initialize_session()
            with open(self.session_file, "r") as f:
                return json.load(f)

    def _save_session_data(self, session_data: Dict) -> None:
        """Save session data to file."""
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def _update_full_history(self) -> None:
        """Update the full history file."""
        try:
            with open(self.full_history_file, "r") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {"sessions": []}

        # Update or add session entry
        session_entry = {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "session_totals": {
                "total_time": 0,
                "total_chunks": 0,
                "total_words": 0
            },
            "interactions": []
        }

        # Check if session already exists in history
        for i, session in enumerate(history["sessions"]):
            if session["session_id"] == self.session_id:
                history["sessions"][i] = session_entry
                break
        else:
            history["sessions"].append(session_entry)

        # Save updated history
        with open(self.full_history_file, "w") as f:
            json.dump(history, f, indent=2)

    def save_interaction(self, prompt: str, model_responses: Dict[str, Any]) -> None:
        """Save an interaction to the session and update all relevant files."""
        timestamp = datetime.now()

        # Create interaction entry
        interaction = {
            "timestamp": timestamp.isoformat(),
            "prompt": prompt,
            "model_responses": {}
        }

        # Add each model's response and timing
        for model, result in model_responses.items():
            interaction["model_responses"][model] = {
                "response": result["response"],
                "time_to_first_token": result["time_to_first_token"],
                "total_time": result["total_time"],
                "chunk_count": result.get("chunk_count", 0),
                "word_count": result.get("word_count", 0)
            }

        # Update session data
        session_data = self._load_session_data()

        # Update session totals
        for model, result in model_responses.items():
            session_data["session_totals"]["total_time"] += result.get("total_time", 0)
            session_data["session_totals"]["total_chunks"] += result.get("chunk_count", 0)
            session_data["session_totals"]["total_words"] += result.get("word_count", 0)

        # Update session benchmarks
        for model, result in model_responses.items():
            if model not in session_data["benchmarks"]["models"]:
                session_data["benchmarks"]["models"][model] = {
                    "avg_total_time": result["total_time"],
                    "avg_first_token": result["time_to_first_token"],
                    "count": 1,
                    "total_chunks": result.get("chunk_count", 0)
                }
            else:
                m = session_data["benchmarks"]["models"][model]
                count = m["count"]
                m["avg_total_time"] = (m["avg_total_time"] * count + result["total_time"]) / (count + 1)
                m["avg_first_token"] = (m["avg_first_token"] * count + result["time_to_first_token"]) / (count + 1)
                m["count"] += 1
                m["total_chunks"] += result.get("chunk_count", 0)

        # Add interaction to session data
        session_data["interactions"].append(interaction)
        self._save_session_data(session_data)

        # Save individual interaction file
        interaction_file = os.path.join(
            self.session_dir,
            f"interaction_{len(session_data['interactions']):04d}.json"
        )
        with open(interaction_file, "w") as f:
            json.dump(interaction, f, indent=2)

        # Update full history
        try:
            with open(self.full_history_file, "r") as f:
                history = json.load(f)

            for session in history["sessions"]:
                if session["session_id"] == self.session_id:
                    session["interactions"].append(interaction)
                    session["session_totals"] = session_data["session_totals"]
                    break

            with open(self.full_history_file, "w") as f:
                json.dump(history, f, indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            self._update_full_history()

    def get_session_info(self) -> Dict:
        """Get session setup information."""
        return {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "contexts_dir": self.contexts_dir,
            "session_file": self.session_file,
            "context_history_file": self.context_history_file,
            "current_context_file": self.current_context_file
        }

    @classmethod
    def get_current_session(cls, base_dir: str = "history") -> Optional[str]:
        """Get the current active session ID."""
        current_session_file = os.path.join(base_dir, "current_session.txt")
        try:
            with open(current_session_file, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    @classmethod
    def create_new_session(cls, base_dir: str = "history") -> 'SessionManager':
        """Create a new session with timestamp-based ID."""
        timestamp = datetime.now()
        session_id = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        return cls(session_id, base_dir)
