import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from miles.utils.ft.platform.feishu_notifier import FeishuWebhookNotifier


class TestFeishuWebhookNotifier:
    @pytest.fixture
    def notifier(self) -> FeishuWebhookNotifier:
        return FeishuWebhookNotifier(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/test-token")

    @pytest.mark.asyncio
    async def test_send_posts_correct_json(self, notifier: FeishuWebhookNotifier) -> None:
        mock_response = httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await notifier.send(title="Fault Alert", content="GPU lost on node-3", severity="critical")

            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert payload["msg_type"] == "interactive"
            assert payload["card"]["header"]["title"]["tag"] == "plain_text"
            assert payload["card"]["header"]["title"]["content"] == "[critical] Fault Alert"
            assert payload["card"]["elements"] == [{"tag": "markdown", "content": "GPU lost on node-3"}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("severity", ["critical", "warning", "info"])
    async def test_send_includes_severity_in_header(
        self, notifier: FeishuWebhookNotifier, severity: str,
    ) -> None:
        mock_response = httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await notifier.send(title="Alert", content="test", severity=severity)

            payload = mock_post.call_args[1]["json"]
            assert payload["card"]["header"]["title"]["content"].startswith(f"[{severity}]")

    @pytest.mark.asyncio
    async def test_send_http_error_does_not_raise(
        self, notifier: FeishuWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with caplog.at_level(logging.WARNING):
                await notifier.send(title="Fault Alert", content="test error", severity="critical")

            assert "feishu_webhook_send_failed" in caplog.text
