import json
import os
import uuid
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

class SessionManager:
    """Manages chat sessions and their persistence according to TRD specifications."""

    def __init__(self, session_dir: str = "history", version: str = "0.1.0"):
        """Initialize the session manager.

        Args:
            session_dir: Directory to store session files
            version: Program version number
        """
        self.session_dir = session_dir
        self.session_id = str(uuid.uuid4())
        self.current_session: List[Dict] = []
        self.version = version

        # Create session directory if it doesn't exist
        os.makedirs(session_dir, exist_ok=True)

        # Initialize session file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(
            session_dir,
            f"session_{self.session_id}_{timestamp}.json"
        )
        self._initialize_session_file()

    def _get_system_metrics(self) -> Dict:
        """Get current system performance metrics."""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "mem_usage_mb": process.memory_info().rss / (1024 * 1024),
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent
        }

    def _get_git_info(self) -> Dict:
        """Get git repository information."""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            return {
                "git_commit": commit,
                "git_branch": branch
            }
        except:
            return {
                "git_commit": "unknown",
                "git_branch": "unknown"
            }

    def _initialize_session_file(self):
        """Initialize the session file with basic structure."""
        session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "program_info": {
                "version": self.version,
                **self._get_git_info()
            }
        }

        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

    def add_interaction(self, interaction: Dict, chain_steps: Optional[List[Dict]] = None, model_details: Optional[Dict] = None):
        """Add an interaction to the current session.

        Args:
            interaction: Dictionary containing interaction details
            chain_steps: Optional list of agent chain steps
            model_details: Optional model configuration details
        """
        # Add required fields if not present
        if "id" not in interaction:
            interaction["id"] = str(uuid.uuid4())
        if "timestamp" not in interaction:
            interaction["timestamp"] = datetime.now().isoformat()

        # Add metadata according to TRD
        interaction["metadata"] = {
            "session_id": self.session_id,
            "resource_usage": self._get_system_metrics(),
            "program_info": {
                "version": self.version,
                **self._get_git_info()
            },
            "custom_tags": []
        }

        # Add model details if provided
        if model_details:
            interaction["metadata"]["model_details"] = model_details

        # Add chain steps if provided
        if chain_steps:
            interaction["chain_steps"] = chain_steps

        self.current_session.append(interaction)
        self._save_interaction(interaction)

    def _save_interaction(self, interaction: Dict):
        """Save an interaction to the session file.

        Args:
            interaction: The interaction to save
        """
        with open(self.session_file, 'r+') as f:
            data = json.load(f)
            data["interactions"].append(interaction)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def get_session_summary(self) -> Dict:
        """Get a summary of the current session.

        Returns:
            Dictionary containing session summary
        """
        return {
            "session_id": self.session_id,
            "session_file": self.session_file,
            "interaction_count": len(self.current_session),
            "start_time": self.current_session[0]["timestamp"] if self.current_session else None,
            "last_interaction": self.current_session[-1] if self.current_session else None,
            "resource_usage": self._get_system_metrics()
        }

    def load_session(self, session_id: str) -> bool:
        """Load a previous session.

        Args:
            session_id: ID of the session to load

        Returns:
            True if session was loaded successfully, False otherwise
        """
        # Find session file
        for filename in os.listdir(self.session_dir):
            if session_id in filename and filename.endswith('.json'):
                filepath = os.path.join(self.session_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.session_id = session_id
                        self.session_file = filepath
                        self.current_session = data["interactions"]
                        return True
                except Exception as e:
                    print(f"Error loading session: {e}")
                    return False
        return False

    def get_recent_interactions(self, limit: int = 5) -> List[Dict]:
        """Get the most recent interactions.

        Args:
            limit: Maximum number of interactions to return

        Returns:
            List of recent interactions
        """
        return self.current_session[-limit:]

    def close_session(self):
        """Close the current session and update metadata."""
        with open(self.session_file, 'r+') as f:
            data = json.load(f)
            data["end_time"] = datetime.now().isoformat()
            data["total_interactions"] = len(self.current_session)
            data["final_resource_usage"] = self._get_system_metrics()
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
