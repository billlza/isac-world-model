"""Contract tests using synthetic polls, never new M7 device measurements.

The synthetic fixture is Mooring's public contract/history-fixtures/radio-samples.csv
at 0d6361d3bca8ea76a08cd8ef1cac6606dd730958 (GPL-3.0-or-later).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from isac_wm.m7_telemetry import (
    COLUMNS,
    RF_FIELDS,
    SAMPLES_HEADER,
    SCHEMA_REFERENCE_COMMIT,
    read_mooring_samples_csv,
    write_telemetry_import,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/m7_mooring_synthetic_samples.csv"


class M7TelemetryContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "synthetic-samples.csv"
        self.input.write_bytes(FIXTURE.read_bytes())
        self.rows = FIXTURE.read_text().splitlines()[1:]

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        self.input.write_text(SAMPLES_HEADER + "\n" + "".join(row + "\n" for row in rows))

    def changed_row(self, index=0, **changes):
        cells = self.rows[index].split(",")
        for name, value in changes.items():
            cells[COLUMNS.index(name)] = value
        return ",".join(cells)

    def test_exact_upstream_synthetic_fixture_and_provenance(self):
        raw = FIXTURE.read_bytes()
        blob_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        self.assertEqual(blob_hash, "a5bcfeb610c85d368c43fe625b734cd56aa1bd63")
        imported = read_mooring_samples_csv(self.input, data_origin="synthetic")
        self.assertEqual(imported["quality"]["records"], 9)
        self.assertEqual(imported["quality"]["poll_outcomes"], {"read": 8, "failed": 1})
        self.assertEqual(imported["quality"]["read_without_service"], 1)
        self.assertEqual(imported["quality"]["attached_reads"], 7)
        self.assertEqual(imported["quality"]["records_with_any_rf_metric"], 7)
        self.assertEqual(imported["quality"]["rows_reordered_or_dropped"], 0)
        source = imported["manifest"]["source"]
        self.assertEqual(source["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(source["size_bytes"], len(raw))
        self.assertEqual(source["schema_reference_commit"], SCHEMA_REFERENCE_COMMIT)
        self.assertIsNone(source["declared_collector_commit"])
        self.assertIsNone(source["declared_firmware"])
        self.assertEqual(source["declared_data_origin"], "synthetic")
        self.assertFalse(source["declarations_verified_by_csv"])
        self.assertIn("unknown", source["upstream_history_completeness"])

    def test_poll_failure_is_distinct_from_reported_no_service(self):
        imported = read_mooring_samples_csv(self.input)
        failed, no_service = imported["records"][1], imported["records"][4]
        self.assertEqual((failed["outcome"], failed["failure"]), ("failed", "unreachable"))
        self.assertIsNone(failed["context"]["service"])
        self.assertEqual(no_service["outcome"], "read")
        self.assertEqual(no_service["context"]["service"], "no_service")
        self.assertIsNone(no_service["failure"])
        self.assertEqual(imported["quality"]["poll_failure_reasons"], {"unreachable": 1})

    def test_missing_mask_preserves_zero_and_plmn_leading_zeros(self):
        self.write_rows([self.changed_row(lte_sinr_db="0.0", nr_rsrp_dbm="", plmn="001-01")])
        row = read_mooring_samples_csv(self.input)["records"][0]
        self.assertEqual(row["rf_metrics"]["lte_sinr_db"], 0.0)
        self.assertFalse(row["rf_missing"]["lte_sinr_db"])
        self.assertIsNone(row["rf_metrics"]["nr_rsrp_dbm"])
        self.assertTrue(row["rf_missing"]["nr_rsrp_dbm"])
        self.assertEqual(row["context"]["plmn"], "001-01")
        self.assertEqual(set(row["rf_metrics"]), set(RF_FIELDS))

    def test_unknown_rf_capture_and_availability_are_not_fabricated(self):
        imported = read_mooring_samples_csv(self.input)
        self.assertEqual(imported["manifest"]["timing"]["timestamp_semantics"], "host_poll_start")
        for row in imported["records"]:
            self.assertIsNone(row["rf_capture_unix_s"])
            self.assertIsNone(row["available_unix_s"])
        capabilities = imported["manifest"]["capabilities"]
        for key in ("online_causal_replay", "calibrated_3d_rf_input", "complex_csi", "raw_iq"):
            self.assertFalse(capabilities[key])
        self.assertTrue(capabilities["offline_scalar_link_telemetry"])

    def test_header_only_export_has_no_usable_measurements(self):
        self.write_rows([])
        imported = read_mooring_samples_csv(self.input)
        self.assertEqual(imported["manifest"]["status"], "empty_export")
        self.assertEqual(imported["records"], [])
        self.assertEqual(imported["quality"]["records_with_any_rf_metric"], 0)
        self.assertIsNone(imported["quality"]["poll_start_intervals_s"]["median"])

    def test_crlf_preserves_source_hash(self):
        self.input.write_bytes(FIXTURE.read_bytes().replace(b"\n", b"\r\n"))
        imported = read_mooring_samples_csv(self.input)
        self.assertEqual(len(imported["records"]), 9)
        self.assertEqual(
            imported["manifest"]["source"]["sha256"],
            hashlib.sha256(self.input.read_bytes()).hexdigest(),
        )

    def test_unknown_header_and_column_counts_fail(self):
        for raw in (
            "timestamp," + SAMPLES_HEADER.split(",", 1)[1] + "\n",
            SAMPLES_HEADER + ",extra\n",
            SAMPLES_HEADER + "\n" + self.rows[0] + ",extra\n",
            SAMPLES_HEADER + "\n" + ",".join(self.rows[0].split(",")[:-1]) + "\n",
        ):
            with self.subTest(raw=raw[:30]):
                self.input.write_text(raw)
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_truncated_or_blank_rows_fail(self):
        for raw in (b"", FIXTURE.read_bytes().rstrip(b"\n"), FIXTURE.read_bytes() + b"\n"):
            with self.subTest(length=len(raw)):
                self.input.write_bytes(raw)
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_timestamp_disagreement_or_invalid_date_fails(self):
        for changes in (
            {"at_unix": "1791918001"},
            {"at_utc": "2026-02-30T19:00:00Z"},
            {"at_utc": "2026-10-13T19:00:00+00:00"},
        ):
            with self.subTest(changes=changes):
                self.write_rows([self.changed_row(**changes)])
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_duplicate_and_out_of_order_rows_fail_without_sorting(self):
        for rows in ([self.rows[0], self.rows[0]], [self.rows[2], self.rows[0]]):
            self.write_rows(rows)
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                read_mooring_samples_csv(self.input)

    def test_nonfinite_noncanonical_or_unrepresentable_rf_fails(self):
        for value in ("NaN", "inf", "-Infinity", "1e3", "2", "3276.8", "-3276.9", " 2.0"):
            with self.subTest(value=value):
                self.write_rows([self.changed_row(nr_sinr_db=value)])
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_all_rf_fields_obey_pinned_value_type_boundaries(self):
        # Independent source-contract cases from Mooring router.rs, in tenths.
        cases = {
            "lte_rsrp_dbm": (-1560, -310),
            "lte_rsrq_db": (-430, 200),
            "lte_sinr_db": (-230, 400),
            "lte_rssi_dbm": (-1500, 0),
            "nr_rsrp_dbm": (-1560, -310),
            "nr_rsrq_db": (-430, 200),
            "nr_sinr_db": (-230, 400),
        }
        for field, (lower, upper) in cases.items():
            for tenths in (lower, lower + 1, upper - 1, upper):
                with self.subTest(field=field, accepted_tenths=tenths):
                    self.write_rows([self.changed_row(**{field: f"{tenths / 10:.1f}"})])
                    record = read_mooring_samples_csv(self.input)["records"][0]
                    self.assertEqual(record["rf_metrics"][field], tenths / 10)
            for tenths in (lower - 1, upper + 1):
                with self.subTest(field=field, rejected_tenths=tenths):
                    self.write_rows([self.changed_row(**{field: f"{tenths / 10:.1f}"})])
                    with self.assertRaisesRegex(ValueError, field):
                        read_mooring_samples_csv(self.input)

    def test_integer_context_fields_obey_pinned_value_type_boundaries(self):
        cases = {
            "temperature_c": (-40, 125),
            "ca_secondary_count": (0, 255),
            "uptime_secs": (0, 2**64 - 1),
            "session_secs": (0, 2**64 - 1),
        }
        for field, (lower, upper) in cases.items():
            for value in (lower, lower + 1, upper - 1, upper):
                with self.subTest(field=field, accepted=value):
                    self.write_rows([self.changed_row(**{field: str(value)})])
                    record = read_mooring_samples_csv(self.input)["records"][0]
                    self.assertEqual(record["context"][field], value)
            for value in (lower - 1, upper + 1):
                with self.subTest(field=field, rejected=value):
                    self.write_rows([self.changed_row(**{field: str(value)})])
                    with self.assertRaisesRegex(ValueError, field):
                        read_mooring_samples_csv(self.input)

    def test_both_band_types_obey_pinned_band_number_boundaries(self):
        for prefix in ("B", "n"):
            for number in (1, 2, 998, 999):
                with self.subTest(prefix=prefix, accepted=number):
                    label = f"{prefix}{number}"
                    self.write_rows([self.changed_row(band=label)])
                    record = read_mooring_samples_csv(self.input)["records"][0]
                    self.assertEqual(record["context"]["band"], label)
            for number in (0, 1000):
                with self.subTest(prefix=prefix, rejected=number):
                    self.write_rows([self.changed_row(band=f"{prefix}{number}")])
                    with self.assertRaisesRegex(ValueError, "band"):
                        read_mooring_samples_csv(self.input)

    def test_reviewed_out_of_contract_values_are_rejected(self):
        for changes in (
            {"lte_rsrp_dbm": "0.0"},
            {"nr_sinr_db": "1000.0"},
            {"temperature_c": "126"},
            {"ca_secondary_count": "256"},
            {"band": "B1000"},
        ):
            with self.subTest(changes=changes):
                self.write_rows([self.changed_row(**changes)])
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_inconsistent_outcomes_and_unknown_enums_fail(self):
        for index, changes in (
            (0, {"failure": "unreachable"}),
            (1, {"lte_rsrp_dbm": "-100.0"}),
            (1, {"failure": "outage"}),
            (4, {"nr_rsrp_dbm": "-100.0"}),
            (0, {"network_service": "wifi"}),
            (0, {"thermal": "made_up"}),
            (0, {"band": "3"}),
        ):
            with self.subTest(index=index, changes=changes):
                self.write_rows([self.changed_row(index, **changes)])
                with self.assertRaises(ValueError):
                    read_mooring_samples_csv(self.input)

    def test_gaps_are_spacing_not_invented_lost_packets(self):
        self.write_rows([self.rows[0], self.rows[-1]])
        quality = read_mooring_samples_csv(self.input)["quality"]
        self.assertEqual(
            quality["gaps_over_2x_nominal"],
            [{"previous_source_row": 2, "source_row": 3, "interval_s": 480}],
        )
        self.assertIn("not a packet-loss estimate", quality["gap_interpretation"])
        self.assertNotIn("lost_packets", quality)

    def test_declared_versions_are_explicit_not_taken_from_schema(self):
        imported = read_mooring_samples_csv(
            self.input,
            collector_commit="a" * 40,
            firmware="test-only-firmware",
            data_origin="synthetic",
        )
        source = imported["manifest"]["source"]
        self.assertEqual(source["declared_collector_commit"], "a" * 40)
        self.assertEqual(source["declared_firmware"], "test-only-firmware")
        for kwargs in ({"data_origin": "measured"}, {"collector_commit": "main"}, {"firmware": "\n"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                read_mooring_samples_csv(self.input, **kwargs)

    def test_output_is_complete_and_existing_directory_is_not_overwritten(self):
        result = read_mooring_samples_csv(self.input, data_origin="synthetic")
        output = self.directory / "imported"
        write_telemetry_import(result, output)
        self.assertEqual(
            set(f.name for f in output.iterdir()),
            {"manifest.json", "quality.json", "telemetry.jsonl"},
        )
        self.assertEqual(len((output / "telemetry.jsonl").read_text().splitlines()), 9)
        self.assertEqual(json.loads((output / "quality.json").read_text()), result["quality"])
        original_manifest = (output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            write_telemetry_import(result, output)
        self.assertEqual((output / "manifest.json").read_bytes(), original_manifest)

    def test_cli_works_without_site_packages_and_fails_before_output_on_bad_input(self):
        output = self.directory / "cli-output"
        command = [
            sys.executable,
            "-S",
            str(ROOT / "scripts/import_m7_telemetry.py"),
            "--input",
            str(self.input),
            "--output",
            str(output),
            "--data-origin",
            "synthetic",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["records"], 9)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["source"]["declared_data_origin"], "synthetic")
        self.input.write_bytes(b"truncated")
        bad_output = self.directory / "must-not-exist"
        command[command.index("--output") + 1] = str(bad_output)
        failed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(failed.returncode, 2)
        self.assertFalse(bad_output.exists())


if __name__ == "__main__":
    unittest.main()
