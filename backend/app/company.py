"""Company identity and dashboard token utilities."""
import hashlib
import re
from typing import Optional

from sqlalchemy.orm import Session

from .settings import COMPANY_SALT, DASHBOARD_TOKEN_SALT


def normalize_company_name(name: str) -> str:
    """Normalize company name for consistent hashing.

    - lowercase
    - trim leading/trailing whitespace
    - collapse multiple whitespace to single space
    - remove punctuation
    """
    # Lowercase and trim
    normalized = name.lower().strip()
    # Collapse multiple whitespace to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized


def compute_company_id(company_name: str) -> str:
    """Compute deterministic company_id from company name.

    Returns: "c_" + sha256(normalize(name) + COMPANY_SALT)[:12]
    """
    normalized = normalize_company_name(company_name)
    hash_input = f"{normalized}{COMPANY_SALT}".encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).hexdigest()
    return f"c_{hash_digest[:12]}"


def compute_dashboard_token(company_id: str) -> str:
    """Compute dashboard token from company_id.

    Returns: sha256(company_id + DASHBOARD_TOKEN_SALT)[:24]
    """
    hash_input = f"{company_id}{DASHBOARD_TOKEN_SALT}".encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).hexdigest()
    return hash_digest[:24]


def resolve_token_to_company_id(token: str, db: Session) -> Optional[str]:
    """Resolve a dashboard token to its company_id.

    Performs an indexed lookup on CompanyRegistry.dashboard_token.
    """
    from .models import CompanyRegistry

    row = db.query(CompanyRegistry.company_id).filter(
        CompanyRegistry.dashboard_token == token
    ).first()

    return row[0] if row else None
