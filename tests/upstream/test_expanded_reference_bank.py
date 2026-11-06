"""Reference-bank ingestion and activation guards, using synthetic CSA only."""
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from build_expanded_reference_bank import (declared_origin, instrument_label, canonical_hostname,
    resolve_metadata, resolve_canonical_outcomes, members, parse_one, parse_global_ini, concat_parquet, ambiguous_acquisition_keys)
from qpm_reference_bank import resolve_reference_bank, file_sha256
from train_multimodal_csa import parse_qpm_stream


class ReferenceIngestTests(unittest.TestCase):
    def test_manual_target_excludes_renamed_export_by_instrument_acquisition(self):
        from run_manual_csa_plugin import manual_target_physical_runs
        core=pd.DataFrame([dict(instrument_id="one",archive_id="different_export",physical_run_id="renamed",run_start_time="2026-10-01T10:00:00Z"),
                           dict(instrument_id="two",archive_id="other",physical_run_id="other_run",run_start_time="2026-10-01T10:00:00Z")])
        record=dict(row=dict(instrument_id="one",archive_id="target",physical_run_id="original",run_start_time="2026-10-01T10:00:00Z"))
        self.assertEqual(manual_target_physical_runs(core,[record]),{"original","renamed"})

    def test_renamed_acquisition_cannot_supply_independent_reference_votes(self):
        old=pd.DataFrame([dict(instrument_id="one",physical_run_id="rawA",run_start_time="2026-10-01T00:00:00Z")])
        results=[dict(run_time_source="raw_run_header_assumed_UTC",row=dict(instrument_id="one",physical_run_id="renamedA",run_start_time="2026-10-01T00:00:00Z"))]
        self.assertEqual(len(ambiguous_acquisition_keys(results,old)),1)
        results[0]["row"]["physical_run_id"]="rawA"
        self.assertFalse(ambiguous_acquisition_keys(results,old))

    def test_parquet_merge_preserves_missing_optional_context_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            pd.DataFrame(dict(archive_id=["a"],value=[1.5],optional=["verified"])).to_parquet(root/"a.parquet",index=False)
            pd.DataFrame(dict(archive_id=["b"],value=[2])).to_parquet(root/"b.parquet",index=False)
            pd.DataFrame().to_parquet(root/"empty.parquet",index=False)
            concat_parquet([root/"a.parquet",root/"b.parquet",root/"empty.parquet"],root/"merged.parquet")
            merged=pd.read_parquet(root/"merged.parquet")
            self.assertEqual(merged.archive_id.tolist(),["a","b"])
            self.assertEqual(merged.value.tolist(),[1.5,2.0])
            self.assertTrue(pd.isna(merged.optional.iloc[1]))

    def test_end_to_end_compact_archive_parse_and_error_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/"input.txz"
            documents={
                "CSA/ion_params_00.json": {"expName":"Auto_R_2026_10_01_10_00_00_ionadmin_GNXS-0073-test", "flows":1000,"chipInfo":{"chipType":"GX7"}},
                "CSA/planned_run.json": {"object":{"runType":"sampleToReport","scriptName":"workflow","resequencing":False}},
                "CSA/analysisSamples.json": {"barcodeIds":["IonDual_0001"],"relations":["DNA"]},
                "CSA/outputs/BaseCallingActor-00/BaseCaller.json": {"Filtering":{"LibraryReport":{"final_library_reads":12345}}},
                "CSA/outputs/BaseCallingActor-00/ionstats_basecaller.json": {"full":{"mean_read_length":180},"Q20":{"mean_read_length":160},"read_group_metrics":{"IonDual_0001":{"read_count":12345}}}}
            with tarfile.open(path,"w:xz") as t:
                for name,value in documents.items():
                    payload=json.dumps(value).encode(); info=tarfile.TarInfo(name);info.size=len(payload);t.addfile(info,BytesIO(payload))
                debug=b"\n".join([b"in PIP.Aspirate id=2 vol=10",b"PipSndV1 id:2,pipCmd:GAid0010ai00100ge1go2lq3", b"pipette recv id = 2, ret = RXid0011rx00010 1 3 5 7",b"pipette response, id = 2, ret = GAid0010er72"])
                info=tarfile.TarInfo("CSA/debug");info.size=len(debug);t.addfile(info,BytesIO(debug))
            record=dict(archive_id="abc",sha256="a"*64,local_path=str(path),container_format="xz_tar",source_instrument_folder="GNXS-0073",filename="run.ChipLane1.CSA.txz",source_path=str(path),size_bytes=path.stat().st_size,source_mtime_utc="2026-10-01T00:00:00Z")
            result=parse_one(record,root,[])
            self.assertEqual(result["status"],"parsed")
            cache=root/".reference_parse_cache/abc"
            full=json.loads((cache/"result.json").read_text())
            self.assertEqual(full["row"]["total_reads"],12345)
            self.assertEqual(full["row"]["mean_read_length_bp"],180)
            self.assertEqual(full["row"]["aq20_mean_read_length_bp"],160)
            self.assertEqual(len(pd.read_parquet(cache/"waves.parquet").qpm_profile.iloc[0]),32)
            self.assertFalse(pd.read_parquet(cache/"quality.parquet").qpm_waveform_eligible.iloc[0])
            self.assertNotIn("sampleNames",(cache/"result.json").read_text())

    def test_cached_template_preserves_physical_run_weight_and_bands(self):
        from qpm_waveform_reference import reference_template, PROFILE_COLUMNS
        rows=[]
        for physical, obs, scale in [("run1","a",1),("run1","a",3),("run1","b",7),("run2","c",9)]:
            rows.append(dict(physical_run_id=physical,observation_id=obs,archive_id=obs,instrument_id="one",
                             run_start_time="2026-10-01T00:00:00Z",**dict(zip(PROFILE_COLUMNS,np.arange(32)*scale))))
        frame=pd.DataFrame(rows)
        expected=frame.groupby(["physical_run_id","observation_id"],sort=False)[PROFILE_COLUMNS].median().groupby(level="physical_run_id",sort=False).median().to_numpy(float)
        template=reference_template(frame,"2026-10-02")
        np.testing.assert_array_equal(template["matrix"],expected)
        np.testing.assert_array_equal(template["median"],np.median(expected,axis=0))
        np.testing.assert_array_equal(template["q05"],np.quantile(expected,.05,axis=0))
        self.assertEqual(len(template["audit_members"]),2)
        self.assertFalse(any(m["reference_includes_future"] for m in template["audit_members"]))
        first=template["audit_members"][0]
        self.assertEqual(first["reference_observation_ids"],"a|b")
        self.assertEqual(first["reference_archive_ids"],"a|b")
        self.assertEqual(first["reference_operation_count"],3)

    def test_sectionless_global_ini_supported(self):
        metadata, _ = resolve_metadata({"CSA/secondary/global.ini": parse_global_ini(b"# config\nanalysis.application.type=DNA_RNA\nir.version.number=7.2\n")})
        self.assertEqual(metadata["global.analysis_application_type"], "DNA_RNA")
        self.assertEqual(metadata["global.ir_version_number"], "7.2")

    def test_boolean_serialization_does_not_split_protocol_peers(self):
        from outcome_reference import normalize
        self.assertEqual(normalize(False), normalize("False"))
        self.assertEqual(normalize(True), normalize("true"))

    def test_origin_header_not_free_text_mention(self):
        self.assertEqual(declared_origin("Auto_R_2026_10_01_10_00_00_ionservice_GNXS-0328-x_from_GNXS-0018"), "gnxs-0328")
        self.assertIsNone(declared_origin("assay_from_GNXS-0328"))
        self.assertEqual(declared_origin("Auto_R_2026_10_01_10_00_00_ionadmin_GNXS-VC06-test"), "gnxs-vc06")

    def test_folder_and_legacy_alias(self):
        self.assertEqual(instrument_label("VC11"), "gnxs-vc11")
        self.assertEqual(instrument_label("GNXS-000B"), "gnxs-000b")
        self.assertEqual(canonical_hostname("gnxs-0083", ["gnxs-0083.example.internal"]), "gnxs-0083.example.internal")

    def test_metadata_name_matches_existing_global_ini_policy(self):
        metadata, warnings = resolve_metadata({"CSA/secondary/global.ini": {"analysis": {"application.type": "DNA_RNA", "regions.file": "/x/panel.bed"}},
                        "CSA/planned_run.json": {"object": {"scriptName": "workflowA", "resequencing": False}},
                        "CSA/ion_params_00.json": {"chipInfo": {"chipType": "GX7"}, "flows": 1000}})
        self.assertEqual(metadata["global.analysis_application_type"], "DNA_RNA")
        self.assertEqual(metadata["ion_params_00.chip_info_chip_type"], "GX7")
        self.assertEqual(metadata["ion_params_00.flows"], "1000")
        self.assertFalse(warnings)

    def test_metadata_conflicts_remain_missing(self):
        metadata, warnings = resolve_metadata({"CSA/secondary/global.ini": {"analysis": {"application.type": "DNA"}},
                                               "CSA/other/global.ini": {"analysis": {"application.type": "RNA"}}})
        self.assertIsNone(metadata["global.analysis_application_type"])
        self.assertIn("global.analysis_application_type", warnings)

    def test_equal_priority_outcome_conflict_is_not_averaged(self):
        vals, provenance = resolve_canonical_outcomes([dict(outcome="total_reads", value=n, priority=1, source_file=f"{n}/Info.csv", source_field="Final Reads") for n in [10, 20]])
        self.assertTrue(np.isnan(vals["total_reads"]))
        self.assertFalse(any(r["selected"] for r in provenance))

    def test_aq20_not_substituted_for_raw_length(self):
        vals, _ = resolve_canonical_outcomes([dict(outcome="aq20_mean_read_length_bp", value=160, priority=0, source_file="ionstats", source_field="Q20.mean_read_length")])
        self.assertTrue(np.isnan(vals["mean_read_length_bp"]))

    def test_compact_waveform_equals_original_and_keeps_er72(self):
        lines = b"\n".join([b"in PIP.Aspirate id=2 vol=10", b"PipSndV1 id:2,pipCmd:GAid0010ai00100ge1go2lq3",
                            b"pipette recv id = 2, ret = RXid0011rx00010 1 3 5 7",
                            b"pipette response, id = 2, ret = GAid0010er72"])
        base = dict(archive_id="test", run_start_time="2026-10-01")
        original = parse_qpm_stream(BytesIO(lines), base)
        stats = {}
        compact = parse_qpm_stream(BytesIO(lines), base, compact=True, quality_stats=stats)
        np.testing.assert_array_equal(compact[0]["qpm_profile"], original[0]["qpm_profile"])
        self.assertFalse(compact[0]["qpm_waveform_eligible"])
        self.assertEqual(stats["er72_operations"], 1)
        self.assertNotIn("qpm_samples", compact[0])

    def test_container_signature_not_extension(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"test.CSA.txz"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("CSA/planned_run.json", "{}")
            values = [(n, h.read()) for n, size, h in members(path, "zip")]
            self.assertEqual(values, [("CSA/planned_run.json", b"{}")])

    def test_corrupt_xz_is_failed_not_empty_normal_archive(self):
        import lzma
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"test.txz"
            payload = BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as t:
                body = b"{}"; info = tarfile.TarInfo("CSA/planned_run.json"); info.size=len(body)
                t.addfile(info, BytesIO(body))
            path.write_bytes(lzma.compress(payload.getvalue())[:-8])
            with self.assertRaises((EOFError, lzma.LZMAError)):
                list(members(path, "xz_tar"))


class ActivationTests(unittest.TestCase):
    def test_explicit_snapshot_unchanged_without_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_reference_bank(d), Path(d).resolve())

    def test_activation_rejects_unvalidated_bank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); bank = root/"bank"; bank.mkdir()
            manifest=bank/"reference_bank_manifest.json"
            manifest.write_text(json.dumps(dict(validation_status="failed", artifacts={})))
            (root/"active_reference_bank.json").write_text(json.dumps(dict(bank_path=str(bank), manifest_sha256=file_sha256(manifest))))
            with self.assertRaises(ValueError): resolve_reference_bank(root)

    def test_activation_rejects_changed_artifact_size(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); bank=root/"bank"; bank.mkdir()
            (bank/"core.parquet").write_bytes(b"x")
            manifest=bank/"reference_bank_manifest.json"
            manifest.write_text(json.dumps(dict(validation_status="passed", artifacts={"core.parquet": {"size_bytes": 2}})))
            (root/"active_reference_bank.json").write_text(json.dumps(dict(bank_path=str(bank), manifest_sha256=file_sha256(manifest))))
            with self.assertRaises(ValueError): resolve_reference_bank(root)


if __name__ == "__main__":
    unittest.main()
