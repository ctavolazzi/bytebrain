import unittest
from unittest.mock import patch, mock_open
from bots.model_comparison import load_config, compare_responses

class TestModelComparison(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "models": ["model1", "model2"],
            "default_prompt": "test prompt",
            "test_prompts": ["prompt1", "prompt2"]
        }

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_config(self, mock_json_load, mock_file):
        """Test config loading"""
        mock_json_load.return_value = self.mock_config
        config = load_config()
        self.assertEqual(config, self.mock_config)
        mock_file.assert_called_once_with('config.json', 'r')

    @patch('bots.ollama_bot.OllamaBot')
    def test_compare_responses(self, mock_bot_class):
        """Test response comparison"""
        # Setup mock bot
        mock_bot = mock_bot_class.return_value
        mock_bot.generate_response.side_effect = [
            "Response from model1",
            "Response from model2"
        ]

        # Capture printed output
        with patch('builtins.print') as mock_print:
            compare_responses("test prompt", ["model1", "model2"])

            # Verify bot was called correctly
            self.assertEqual(mock_bot.generate_response.call_count, 2)
            mock_bot.generate_response.assert_any_call("test prompt", model="model1")
            mock_bot.generate_response.assert_any_call("test prompt", model="model2")

            # Verify output was printed
            mock_print.assert_any_call("\nPrompt: test prompt\n")
            mock_print.assert_any_call("=== model1 ===")
            mock_print.assert_any_call("=== model2 ===")

    @patch('bots.ollama_bot.OllamaBot')
    def test_error_handling(self, mock_bot_class):
        """Test handling of failed responses"""
        # Setup mock bot to return None (error case)
        mock_bot = mock_bot_class.return_value
        mock_bot.generate_response.return_value = None

        # Capture printed output
        with patch('builtins.print') as mock_print:
            compare_responses("test prompt", ["model1"])
            mock_print.assert_any_call("Error: No response received\n")

if __name__ == '__main__':
    unittest.main()
