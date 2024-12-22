import unittest
from unittest.mock import Mock, patch
from bots.ollama_bot import OllamaBot

class TestOllamaBot(unittest.TestCase):
    def setUp(self):
        self.bot = OllamaBot(benchmark=False)
        self.test_prompt = "Hi"

    def test_init(self):
        """Test initialization"""
        self.assertIsNotNone(self.bot.client)

    def test_generate_response_basic(self):
        """Test basic response generation"""
        with patch.object(self.bot.client, 'chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test response'}}
            response = self.bot.generate_response(self.test_prompt)
            self.assertEqual(response, 'Test response')
            mock_chat.assert_called_with(
                model="llama3.2",  # Default model
                messages=[{'role': 'user', 'content': self.test_prompt}],
                stream=False
            )

    def test_generate_response_custom_model(self):
        """Test response with custom model"""
        with patch.object(self.bot.client, 'chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test response'}}
            response = self.bot.generate_response(self.test_prompt, model="codellama")
            self.assertEqual(response, 'Test response')
            mock_chat.assert_called_with(
                model="codellama",
                messages=[{'role': 'user', 'content': self.test_prompt}],
                stream=False
            )

    def test_generate_response_with_role(self):
        """Test response with different role"""
        with patch.object(self.bot.client, 'chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test response'}}
            response = self.bot.generate_response(self.test_prompt, role='system')
            self.assertEqual(response, 'Test response')
            mock_chat.assert_called_with(
                model="llama3.2",
                messages=[{'role': 'system', 'content': self.test_prompt}],
                stream=False
            )

    def test_generate_response_streaming(self):
        """Test streaming response"""
        mock_stream = [
            {'message': {'content': 'chunk1'}},
            {'message': {'content': 'chunk2'}}
        ]
        with patch.object(self.bot.client, 'chat') as mock_chat:
            mock_chat.return_value = mock_stream
            stream = self.bot.generate_response(self.test_prompt, stream=True)
            self.assertEqual(list(stream), mock_stream)
            mock_chat.assert_called_with(
                model="llama3.2",
                messages=[{'role': 'user', 'content': self.test_prompt}],
                stream=True
            )

    def test_error_handling(self):
        """Test error handling"""
        with patch.object(self.bot.client, 'chat', side_effect=Exception("Test error")):
            response = self.bot.generate_response(self.test_prompt)
            self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()