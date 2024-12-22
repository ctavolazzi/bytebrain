import unittest
from unittest.mock import patch, Mock
from ollama_client import OllamaClient

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()
        self.test_messages = [{'role': 'user', 'content': 'Hello'}]

    def test_chat_streaming(self):
        """Test streaming chat response"""
        mock_response = [
            {'message': {'content': 'Hello'}},
            {'message': {'content': ' world'}}
        ]
        with patch('ollama.Client') as mock_client:
            mock_client.return_value.chat.return_value = mock_response
            response = self.client.chat('llama2', self.test_messages, stream=True)
            self.assertEqual(list(response), mock_response)

    def test_chat_non_streaming(self):
        """Test non-streaming chat response"""
        mock_response = {'message': {'content': 'Hello world'}}
        with patch('ollama.Client') as mock_client:
            mock_client.return_value.chat.return_value = mock_response
            response = self.client.chat('llama2', self.test_messages, stream=False)
            self.assertEqual(response, mock_response)

    def test_chat_error(self):
        """Test error handling in chat"""
        with patch('ollama.Client') as mock_client:
            mock_client.return_value.chat.side_effect = Exception("Test error")
            with self.assertRaises(Exception):
                self.client.chat('llama2', self.test_messages)

if __name__ == '__main__':
    unittest.main()
