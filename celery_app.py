from celery import Celery

from config import Config


celery_app = Celery(
    "ai_automation_agent",
    broker=Config.CELERY_BROKER_URL,
    # Ensure the module that defines @celery_app.task is imported,
    # so tasks like "whatsapp_mcp.process_message" are registered.
    include=["blueprints.whatsapp_assistant_mcp"],
)

# We only use Celery for fire-and-forget side‑effectful tasks (Twilio sends),
# so we don't need to persist results.
celery_app.conf.update(
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
)


