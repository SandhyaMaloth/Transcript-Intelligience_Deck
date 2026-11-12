"""Stage 1 - call type (support / external / internal).

Call type is a structural property (who is on the call and how it was booked),
not a semantic one, so deterministic rules on metadata are used: free, exact
and auditable.

Rules, in priority order:
  1. No external email domain                         -> internal
  2. Title starts "Support Case #", "URGENT:" or
     "ESCALATION:"                                    -> support
  3. Any other call with a customer                   -> external (account management)

Edge case: "Aegis / Northstar Pharma - Urgent: Detect Outage Impact" is urgent
but booked by the CSM in the account-management format, so it is `external`;
"ESCALATION: Northstar Pharma ..." was run by support and is `support`.
"""
import re

import pandas as pd

SUPPORT_TITLE = re.compile(r"^(support case\s*#|urgent:|escalation:)", re.I)


def classify_call_type(row: pd.Series) -> str:
    if not row["external_domains"]:
        return "internal"
    if SUPPORT_TITLE.search(row["title"].strip()):
        return "support"
    return "external"


def add_call_type(meetings: pd.DataFrame) -> pd.DataFrame:
    return meetings.assign(call_type=meetings.apply(classify_call_type, axis=1))
