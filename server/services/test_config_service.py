import unittest

from services.config_service import DEFAULT_PROVIDERS_CONFIG


class DefaultProvidersConfigTest(unittest.TestCase):
    def test_atlascloud_provider_uses_openai_compatible_defaults(self):
        provider = DEFAULT_PROVIDERS_CONFIG['atlascloud']

        self.assertEqual(provider['url'], 'https://api.atlascloud.ai/v1/')
        self.assertEqual(provider['api_key'], '')
        self.assertEqual(provider['max_tokens'], 8192)
        self.assertEqual(
            provider['models'],
            {'deepseek-ai/deepseek-v4-pro': {'type': 'text'}},
        )


if __name__ == '__main__':
    unittest.main()
