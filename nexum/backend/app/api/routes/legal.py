"""
Nexum Legal API — Privacy Policy metadata and acknowledgment.
"""
from __future__ import annotations

from datetime import date
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/legal", tags=["legal"])

EFFECTIVE_DATE = "2026-02-11"
POLICY_VERSION = "1.0.0"


class PolicySection(BaseModel):
    number: int
    title: str
    anchor: str


class PrivacyPolicyMeta(BaseModel):
    version: str
    effective_date: str
    last_updated: str
    contact_email: str
    removal_email: str
    sections: List[PolicySection]
    youtube_tos_url: str
    google_privacy_url: str
    google_revoke_url: str


SECTIONS = [
    PolicySection(number=1, title="Introduction", anchor="section-1"),
    PolicySection(number=2, title="Information We Collect", anchor="section-2"),
    PolicySection(number=3, title="Public vs. Private Data Clarification", anchor="section-3"),
    PolicySection(number=4, title="How Information Is Used", anchor="section-4"),
    PolicySection(number=5, title="Data Retention", anchor="section-5"),
    PolicySection(number=6, title="Data Deletion and User Rights", anchor="section-6"),
    PolicySection(number=7, title="Cookies and Analytics", anchor="section-7"),
    PolicySection(number=8, title="Third-Party Services", anchor="section-8"),
    PolicySection(number=9, title="Security", anchor="section-9"),
    PolicySection(number=10, title="Children's Privacy", anchor="section-10"),
    PolicySection(number=11, title="Changes to This Policy", anchor="section-11"),
    PolicySection(number=12, title="Contact Information", anchor="section-12"),
]


@router.get("/privacy", response_model=PrivacyPolicyMeta)
async def get_privacy_policy_metadata():
    """
    Returns privacy policy metadata: version, effective date, section index,
    and compliance-relevant URLs. The full policy text is rendered client-side
    at /privacy.
    """
    return PrivacyPolicyMeta(
        version=POLICY_VERSION,
        effective_date=EFFECTIVE_DATE,
        last_updated=EFFECTIVE_DATE,
        contact_email="privacy@nexum.app",
        removal_email="removal@nexum.app",
        sections=SECTIONS,
        youtube_tos_url="https://www.youtube.com/t/terms",
        google_privacy_url="https://policies.google.com/privacy",
        google_revoke_url="https://security.google.com/settings/security/permissions",
    )


@router.get("/privacy/version")
async def get_privacy_policy_version():
    """Quick version check for cache-busting or consent re-prompts."""
    return {
        "version": POLICY_VERSION,
        "effective_date": EFFECTIVE_DATE,
    }
