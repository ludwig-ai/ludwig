"""Functions to extract engineered features from email data."""
import csv
import logging
import re
from html.parser import HTMLParser
from io import StringIO
from string import whitespace
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """HTML parser to extract body text and ignore HTML markup."""

    def __init__(self):
        HTMLParser.__init__(self)
        self._text = []

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = re.sub("[ \t\r\n]+", " ", text)
            self._text.append(text + " ")

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._text.append("\n\n")
        elif tag == "br":
            self._text.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._text.append("\n\n")

    def text(self):
        return "".join(self._text).strip()


def text_from_html(html_or_text):
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_or_text)
    except Exception as e:
        logger.warning("HTMLParser raised an exception: ", str(e))
        return html_or_text
    body_text = parser.text()
    if not body_text:
        return html_or_text
    return body_text


def get_header_sequence(message):
    # message is an ordered dict of header fields.
    return list(message.keys())


def strip_name_from_address(address):
    """Strips name and brackets from email address, returns user@domain."""
    if address is None:
        return None
    result = re.search(r"<(.+)>", address)
    if result:
        return result.group(1).strip(whitespace + "\"'")
    else:
        return address.strip(whitespace + "\"'")


def get_domain(address):
    if address and "@" in address:
        return strip_name_from_address(address.lower()).split("@")[-1]
    else:
        return None


def _parse_comma_separated_list(text: str) -> List[str]:
    """Parses a comma-separated list, splits on commas unless the comma is inside double-quotes."""
    s = StringIO(text)
    reader = csv.reader(s, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
    try:
        return next(reader)
    except StopIteration:
        return []


def features_from_message(extracted_fields, message):
    """Returns engineered features.

    Args:
      extracted_fields (dict) Columns extracted from message.
      message (email.message.EmailMessage) The original message.
    Return: A dictionary of extracted features.
    """
    # Split to field into a list, ignoring commas inside quotes.
    receiver_list = _parse_comma_separated_list(extracted_fields["to"])
    cc_list = _parse_comma_separated_list(extracted_fields["cc"])
    # Strip name, brackets, and whitespace from each address.
    receiver_emails = [strip_name_from_address(a) for a in receiver_list]
    cc_emails = [strip_name_from_address(a) for a in cc_list]
    # Remove any entries that are '' or None
    receiver_emails = [e for e in receiver_emails if e is not None and len(e) > 0]
    cc_emails = [e for e in cc_emails if e is not None and len(e) > 0]

    date = extracted_fields["date"]
    dt = None
    try:
        dt = pd.to_datetime(date)
    except Exception as e:
        logger.exception(f"Exception thrown reading date: {date}", e)
    if dt:
        hours_since_midnight = (dt - dt.normalize()) / pd.Timedelta("1 hour")
    else:
        hours_since_midnight = 0  # Failed to parse date.

    sender_address = strip_name_from_address(extracted_fields["from"])
    sender_domain = get_domain(sender_address)
    receiver_domains = [get_domain(a) for a in receiver_emails]
    subject = extracted_fields["subject"]
    body = text_from_html(extracted_fields["body"])
    return {
        "sender_address": " ".join(sender_address),
        "receiver_addresses": " ".join(receiver_emails),
        "receiver_address": receiver_emails[0] if len(receiver_emails) == 1 else "",
        "cc_addresses": " ".join(cc_emails),
        "internal_communication": all(sender_domain == rd for rd in receiver_domains),
        "external_communication": any(sender_domain != rd for rd in receiver_domains),
        "n_recipients": len(receiver_list),
        "n_cc": len(cc_list),
        "header_sequence": " ".join(get_header_sequence(message)),
        "content": f"Subject: {subject}\n\n{body}",
        "time_of_day": hours_since_midnight,
    }
