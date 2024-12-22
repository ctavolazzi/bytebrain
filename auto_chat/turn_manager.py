from typing import List, Optional

class TurnManager:
    def __init__(self, models: List[str]):
        """Initialize the turn manager with a list of model names."""
        self.models = models
        self.current_index = 0
        self.context_manager_turn = False

    def get_current(self) -> Optional[str]:
        """Get the current model's name or None if it's context manager's turn."""
        if self.context_manager_turn:
            return None
        if self.current_index >= len(self.models):
            self.current_index = 0
        return self.models[self.current_index]

    def advance(self) -> None:
        """Advance to the next turn, alternating between models and context manager."""
        if self.context_manager_turn:
            self.context_manager_turn = False
            return

        self.current_index = (self.current_index + 1) % len(self.models)
        if self.current_index == 0:  # Completed a full cycle of models
            self.context_manager_turn = True

    def is_context_manager_turn(self) -> bool:
        """Check if it's the context manager's turn."""
        return self.context_manager_turn
