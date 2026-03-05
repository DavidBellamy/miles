from typing import Any

from miles.utils.ft.platform.notifiers.webhook_notifier import WebhookNotifier


class LarkWebhookNotifier(WebhookNotifier):
    """Sends notifications via Lark custom bot webhook (interactive card)."""

    def _build_payload(self, title: str, content: str, severity: str) -> dict[str, Any]:
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"[{severity}] {title}",
                    }
                },
                "elements": [{"tag": "markdown", "content": content}],
            },
        }
