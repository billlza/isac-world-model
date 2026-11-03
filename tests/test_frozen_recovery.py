"""Frozen recovery interventions: causal information sets, unmodified replay, and mask semantics."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from frozen_recovery import (
    anchor_memory,
    counterfactual_snapshots,
    evaluate_case,
    factual_replay_check,
    reference,
)
from recovery_challenge_core import observed_trace, score
from score_scale_recovery import apply_network

from isac_wm.simulator import SceneConfig


class FrozenRecoveryTests(unittest.TestCase):
    def test_camera_mute_keeps_cached_image_and_rf_chronology(self):
        cfg = SceneConfig(steps=64)
        snaps, ready, _, meta = observed_trace(83004500, cfg, True)
        pivot = meta["first_post_blackout_camera_delivery_tick"]
        muted, removed = counterfactual_snapshots(cfg, ready, snaps, pivot)
        self.assertGreater(removed, 0)
        self.assertEqual(snaps[pivot]["new"][0], 1)
        self.assertTrue(all(s["new"][0] == 0 for s in muted[pivot:]))
        for i, case in enumerate(muted):
            np.testing.assert_array_equal(case["radio"], snaps[i]["radio"])
            if i >= pivot:
                np.testing.assert_array_equal(case["image"], snaps[pivot-1]["image"])
            else:
                for key in case:
                    np.testing.assert_array_equal(case[key], snaps[i][key])
        self.assertGreater(muted[pivot]["age"][0], snaps[pivot]["age"][0])

    def test_reference_digest_and_individual_point_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"recovery-96.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "archived"):
                reference(root,7,96)
        original = {"post_errors_m": [0.25]*11, "position_error_auc_m_s":0.25,
                    "blackout_rmse_m":0.3,"three_step_forecast_rmse_m":0.25,
                    "not_recovered_by_1s":False, "recovery_confirmed_s":0.2}
        modified = original | {"post_errors_m": [0.30,0.225]+[0.25]*9}
        with self.assertRaisesRegex(ValueError,"reproduce"):
            factual_replay_check(modified,original)

    def test_real_checkpoint_camera_by_memory_and_retained_state(self):
        import torch

        from isac_wm.artifacts import model_fingerprint, save_checkpoint
        from isac_wm.structured import StructuredConfig, build_model
        from isac_wm.training import load_model

        cfg=SceneConfig(steps=64)
        conf=StructuredConfig(hidden_dim=96, architecture="structured-attention")
        original_threads=torch.get_num_threads()
        self.addCleanup(torch.set_num_threads,original_threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model=build_model(conf).eval()
        mem=model.initial(1,"cpu")
        mem[7].fill_(True)
        masked=anchor_memory(mem,mask_old=True)
        self.assertTrue(mem[7].all())
        self.assertFalse(masked[7].any())
        for i in (0,1,2,3,4,5,6,8):
            torch.testing.assert_close(masked[i],mem[i],rtol=0,atol=0)
        episode_seed=83004500
        snaps, _, truth, meta=observed_trace(episode_seed,cfg,True)
        anchor=meta["first_post_blackout_camera_delivery_tick"]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"model.pt"
            save_checkpoint(path, {"schema_version":2, "model_config":conf.to_dict(),
                                 "scene_config":cfg.to_dict(),"state_dict":model.state_dict(),
                                 "train_episode_seeds":[123],"validation_episode_seeds":[124]})
            loaded, cfg, dev=load_model(path,"cpu")
            before=model_fingerprint(loaded)
            xy,fc=apply_network(loaded,cfg,dev,snaps)
            base=score(xy,truth[:,:2],anchor)
            base["three_step_forecast_rmse_m"]=float(np.sqrt(np.mean(np.sum((fc[5:-3]-truth[8:,:2])**2,axis=1))))
            ref={"observations":meta,"scores":{"structured-attention":base}}
            # Inspect the actual MHA padding mask when a retained null key and new evidence arrive.
            masks=[]
            def hook(module,args,kwargs):
                masks.append(kwargs["key_padding_mask"].detach().clone())
            handle=loaded.attention.register_forward_pre_hook(hook,with_kwargs=True)
            try:
                r=evaluate_case(loaded,cfg,dev,episode_seed,"maneuver",ref)
            finally:
                handle.remove()
            self.assertEqual(before,model_fingerprint(loaded))
        cells=r["cells"]
        self.assertEqual(len(cells),4)
        self.assertEqual(len({v["prearrival_error_m"] for v in cells.values()}),1)
        self.assertLessEqual(r["factual_curve_max_abs_deviation_m"],1e-5)
        self.assertGreater(r["preanchor_occupied_event_slots"],0)
        first_masked=anchor+(len(snaps)-anchor)
        mask=masks[first_masked][0]
        self.assertFalse(mask[0].item())  # null stays visible
        self.assertEqual(int((~mask[1:]).sum()),int(snaps[anchor]["new"].sum()))
        self.assertFalse(np.allclose(cells["camera_on_oldkeys_on"]["post_errors_m"],
                                     cells["camera_off_oldkeys_on"]["post_errors_m"]))
