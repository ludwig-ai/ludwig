"""Functions to parse email from text files."""
import email.parser
import email.policy
import logging

logger = logging.getLogger(__name__)


def strip_brackets(s):
    """Removes < and > from e-mail address or e-mail ID."""
    if s is None:
        return None
    s = s.strip()
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1]
    return s


def read_email(path):
    """Reads email from a file, returns a email.message.EmailMessage."""
    with open(path, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def message_body_text(message):
    """Extract body text from message.

    Returns the text/plain or text/html body
    """
    # Most messages are multipart with a text/plain or text/html part.
    try:
        body_part = message.get_body()
    except Exception as e:
        logger.exception("Exception thrown while parsing email message:", e)
        return None
    if body_part is None:
        return None
    elif body_part.get_content_maintype() == "multipart":
        for part in body_part.iter_parts():
            if part.get_content_maintype() == "text":
                return part.get_content()
    else:
        try:
            return body_part.get_content()
        except Exception as e:
            logger.exception("Exception thrown while parsing email message:", e)
            return None


def message_to_columns(message, label):
    """Converts email message to a dict of key/value pairs."""

    def safe_get_field(m, f, default=None):
        try:
            return strip_brackets(m.get(f, default))
        except Exception as e:
            logger.exception("Exception thrown while parsing email field:", e)
            return default

    return {
        "label": label,
        "message_id": safe_get_field(message, "Message-ID"),
        "date": safe_get_field(message, "Date"),
        "from": safe_get_field(message, "From", ""),
        "to": safe_get_field(message, "To", ""),
        "cc": safe_get_field(message, "Cc", ""),
        "bcc": safe_get_field(message, "Bcc", ""),
        "subject": safe_get_field(message, "Subject", ""),
        "body": message_body_text(message),
        "x_mailer": safe_get_field(message, "X-Mailer"),
    }
