"""Regression and unit tests. Run from the repo root: python -m unittest discover tests"""
import os
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ti import discovery, insights  # noqa: E402
from ti.config import SAMPLE_DIR  # noqa: E402
from ti.llm_labeler import validate_label  # noqa: E402
from ti.load import match_speaker_email  # noqa: E402
from ti.pipeline import run  # noqa: E402
from ti.taxonomy import keyword_classify  # noqa: E402


class TestFullDataset(unittest.TestCase):
    """Pins the headline numbers reported in the README, notebook and deck."""

    @classmethod
    def setUpClass(cls):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        cls.calls, cls.utts = run()

    def test_shape_and_labels(self):
        self.assertEqual(len(self.calls), 100)
        self.assertEqual(self.calls.call_type.value_counts().to_dict(), {"external": 40, "support": 30, "internal": 30})
        self.assertTrue((self.calls.label_source == "llm").all())
        self.assertTrue(self.utts.email.notna().all(), "every speaker should match an email")

    def test_headline_numbers(self):
        c = self.calls
        self.assertAlmostEqual((c.kw_primary == c.primary).mean(), 0.80)
        sq = insights.support_quality(c)
        self.assertAlmostEqual(sq["aegis_root_cause_pct"], 0.70, places=2)
        self.assertAlmostEqual(sq["avg_customer_net"], -0.37, places=2)
        self.assertAlmostEqual(sq["avg_aegis_net"], 0.01, places=2)
        self.assertEqual(insights.early_warning_signals(self.utts, c).lead_days.max(), 32)
        self.assertEqual(len(insights.remediation_aftershocks(c)), 5)

    def test_competitor_shift(self):
        cm = insights.competitor_mentions(self.utts, self.calls)
        ss = cm[cm.competitor == "SentinelShield"].groupby("post_outage").meeting_id.nunique()
        self.assertEqual((ss[False], ss[True]), (3, 19))

    def test_clustering(self):
        labels, k, _, _ = discovery.cluster(self.calls)
        self.assertEqual(k, 11)
        self.assertAlmostEqual(discovery.agreement(labels, self.calls.primary), 0.38, places=2)


class TestNewCall(unittest.TestCase):
    """A call that is not in the label cache must still flow through the pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.calls, cls.utts = run(SAMPLE_DIR)

    def test_sample_is_scored_with_keyword_fallback(self):
        r = self.calls.iloc[0]
        self.assertEqual(r.call_type, "support")
        self.assertEqual(r.label_source, "keyword")
        self.assertEqual(r.primary, "DEFECT")
        self.assertEqual(r.competitors, ["SentinelShield"])
        self.assertLess(r.customer_net, 0)
        self.assertGreater(r.recovery, 0)

    def test_sample_speaker_sides(self):
        sides = self.utts.groupby("speaker").side.first().to_dict()
        self.assertEqual(sides, {"Priya Patel": "aegis", "Owen Price": "customer"})


class TestApiPath(unittest.TestCase):
    """Exercises label_with_claude against a fake `anthropic` module (no network, no cost)."""

    def test_request_and_parsing(self):
        import json
        import types
        from unittest import mock

        from ti import llm_labeler
        from ti.call_type import add_call_type
        from ti.load import load_dataset

        label = {"primary": "DEFECT", "secondary": ["RENEWAL"], "churn_risk": "high", "outage_related": True,
                 "comm_gap": True, "support_delay": False, "known_issue": True, "remediation_side_effect": False,
                 "root_cause": "aegis", "resolution": "workaround_provided", "rationale": "Known 4.3.0 bug."}
        calls = []

        class FakeMessages:
            def create(self, **kwargs):
                calls.append(kwargs)
                block = types.SimpleNamespace(type="text", text=json.dumps(label))
                return types.SimpleNamespace(stop_reason="end_turn", content=[block])

        fake = types.ModuleType("anthropic")
        fake.Anthropic = lambda **_: types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages()))
        fake.AuthenticationError = fake.APIConnectionError = type("E", (Exception,), {})

        m, u = load_dataset(SAMPLE_DIR)
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            out = llm_labeler.label_with_claude(add_call_type(m), u)

        self.assertEqual(out, {"01SAMPLE0000HARBORLANE001": label})
        req = calls[0]
        self.assertNotIn("temperature", req)
        self.assertEqual(req["output_config"]["format"]["schema"], llm_labeler.LABEL_SCHEMA)
        self.assertIn("[customer] Owen Price:", req["messages"][0]["content"])


class TestUnits(unittest.TestCase):
    def test_speaker_matching(self):
        emails = ["kevin.obrien@aegiscloud.com", "o.price@harborlanefoods.com", "j.lee@x.com", "fleetops@x.com"]
        self.assertEqual(match_speaker_email("Kevin O'Brien", emails), "kevin.obrien@aegiscloud.com")
        self.assertEqual(match_speaker_email("Owen Price", emails), "o.price@harborlanefoods.com")
        self.assertEqual(match_speaker_email("Jane Lee", emails), "j.lee@x.com")
        self.assertIsNone(match_speaker_email("Sam Smith", emails))

    def test_keyword_default_respects_call_type(self):
        self.assertEqual(keyword_classify("nothing relevant", "internal"), "PLANNING")
        self.assertEqual(keyword_classify("nothing relevant", "support"), "DEFECT")
        self.assertEqual(keyword_classify("nothing relevant", "external"), "RENEWAL")
        self.assertNotIn(keyword_classify("sprint retro roadmap", "external"), {"PLANNING", "LAUNCH", "GTM"})

    def test_label_validation(self):
        good = {"primary": "DEFECT", "secondary": ["RENEWAL"], "churn_risk": "high", "outage_related": True,
                "comm_gap": True, "support_delay": False, "known_issue": True, "remediation_side_effect": False,
                "root_cause": "aegis", "resolution": "workaround_provided", "rationale": "Known 4.3.0 bug."}
        validate_label("ok", good)
        for bad in ({"primary": "OTHER"}, {"secondary": ["DEFECT"]}, {"secondary": ["A", "B", "C"]},
                    {"comm_gap": "yes"}, {"rationale": ""}):
            with self.assertRaises(ValueError):
                validate_label("bad", {**good, **bad})


if __name__ == "__main__":
    unittest.main()
