"""Local end-to-end inference tests (3 local test studies + synthetic edge cases)."""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


def main():
    W = "D:/rsna_work/tmp/infer_test"
    ckpt_src = sys.argv[1] if len(sys.argv) > 1 else f"{R}/output/imaging_v5/runs/smoke/ckpt_best.pt"
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(f"{W}/models")
    shutil.copy(ckpt_src, f"{W}/models/model_a.pt")
    dev = os.environ.get("KNEE_TEST_DEVICE", "cuda")          # "cpu" while a training run occupies the GPU
    base = dict(test_csv=f"{R}/data/Raw_dicom/test.csv", sample_submission=f"{R}/data/Raw_dicom/sample_submission.csv",
                model_dir=f"{W}/models", workers=2, batch_size=2, device=dev)

    # 1) normal run on the local test stub
    cfg = dict(base, test_series_csv=f"{R}/data/Raw_dicom/test_series.csv", test_dicom_root=f"{R}/data/Raw_dicom/test_series",
               cache_dir=f"{W}/cache1", out_path=f"{W}/sub1/submission.csv")
    os.makedirs(f"{W}/sub1")
    sub, rep = predict.run(cfg)
    print("TEST 1 ok:", sub.shape, "| decode failures:", len(rep["decode_failures"]))

    # 2) corrupt one FS series (truncate every file) -> must fall back / mask, never zero-fill
    ts = pd.read_csv(f"{R}/data/Raw_dicom/test_series.csv")
    shutil.copytree(f"{R}/data/Raw_dicom/test_series", f"{W}/dicom2")
    victim = ts[(ts.Fluid_Sensitive == 1) & (ts.Anatomical_Plane == "Sagittal")].iloc[0]
    vdir = f"{W}/dicom2/{victim.StudyInstanceUID}/{victim.SeriesInstanceUID}"
    for f in os.listdir(vdir):
        p = f"{vdir}/{f}"
        data = open(p, "rb").read()
        open(p, "wb").write(data[: len(data) - 20000])        # header intact, pixel data truncated
    cfg2 = dict(cfg, test_dicom_root=f"{W}/dicom2", cache_dir=f"{W}/cache2", out_path=f"{W}/sub2/submission.csv")
    os.makedirs(f"{W}/sub2")
    predict.MAX_FAILED_STUDY_FRACTION = 1.0                   # 1/3 studies affected in this tiny test set
    sub2, rep2 = predict.run(cfg2)
    print("TEST 2 ok: decode failures logged:", [(f["slot"], f["error"][:60]) for f in rep2["decode_failures"]])

    # 3) a study whose coronal series are all absent -> slot masked, prediction still produced
    ts3 = ts[~((ts.StudyInstanceUID == ts.StudyInstanceUID.iloc[0]) & (ts.Anatomical_Plane == "Coronal"))]
    ts3.to_csv(f"{W}/test_series_nocor.csv", index=False)
    cfg3 = dict(cfg, test_series_csv=f"{W}/test_series_nocor.csv", cache_dir=f"{W}/cache3", out_path=f"{W}/sub3/submission.csv")
    os.makedirs(f"{W}/sub3")
    sub3, rep3 = predict.run(cfg3)
    print("TEST 3 ok:", sub3.shape)
    d = np.abs(sub3.iloc[:, 1:].values - sub.iloc[:, 1:].values).max(axis=1)
    print("max |pred change| per study when coronal removed from study 0:", np.round(d, 4))

    # 5) mixed-preprocessing ensemble: a v2 and a v6 checkpoint -> both caches built (224 / 256 px),
    #    ensemble == mean of the single-model runs
    import torch
    ck = torch.load(ckpt_src, map_location="cpu", weights_only=False)
    ck["args"] = dict(ck["args"], preproc="v6")
    for sub_dir, files in (("m_v2", {"model_a.pt": None}), ("m_v6", {"model_b.pt": ck}), ("m_mix", {"model_a.pt": None, "model_b.pt": ck})):
        os.makedirs(f"{W}/{sub_dir}")
        for fn, obj in files.items():
            if obj is None:
                shutil.copy(ckpt_src, f"{W}/{sub_dir}/{fn}")
            else:
                torch.save(obj, f"{W}/{sub_dir}/{fn}")
    res = {}
    for sub_dir in ("m_v2", "m_v6", "m_mix"):
        os.makedirs(f"{W}/sub5_{sub_dir}")
        c5 = dict(cfg, model_dir=f"{W}/{sub_dir}", cache_dir=f"{W}/cache5", out_path=f"{W}/sub5_{sub_dir}/submission.csv")
        res[sub_dir], rep5 = predict.run(c5)
    shapes = {pp_: sorted({np.load(f"{W}/cache5/{pp_}/{f}", mmap_mode="r").shape[1:] for f in os.listdir(f"{W}/cache5/{pp_}")})
              for pp_ in ("v2", "v6")}
    mix = res["m_mix"].iloc[:, 1:].values
    mean = (res["m_v2"].iloc[:, 1:].values + res["m_v6"].iloc[:, 1:].values) / 2
    diff_v2_v6 = np.abs(res["m_v2"].iloc[:, 1:].values - res["m_v6"].iloc[:, 1:].values).max()
    assert shapes == {"v2": [(224, 224)], "v6": [(256, 256)]}, shapes
    assert np.allclose(mix, mean, atol=1e-6), np.abs(mix - mean).max()
    print("TEST 5 ok: cache shapes", shapes, "| ensemble == mean of single models | max |v2-v6| pred diff", round(float(diff_v2_v6), 4),
          "| report preproc_versions", rep5["preproc_versions"])

    # 6) end-slice policy (infer_lib.decode_series): one truncated slice at the END of the slice order is
    #    dropped and reported (series still used, T-1 slices); one truncated MIDDLE slice fails the series
    from infer_lib import order_series_files
    big = ts.assign(n=ts.apply(lambda r: len(os.listdir(f"{R}/data/Raw_dicom/test_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}")), axis=1))
    big = big[(big.Fluid_Sensitive == 1) & (big.n >= 10)].iloc[0]
    for case in ("end", "middle"):
        root = f"{W}/dicom6_{case}"
        shutil.copytree(f"{R}/data/Raw_dicom/test_series", root)
        sdir = f"{root}/{big.StudyInstanceUID}/{big.SeriesInstanceUID}"
        order = [f for f, _ in order_series_files(sdir)]
        victim = order[-1] if case == "end" else order[len(order) // 2]
        data = open(f"{sdir}/{victim}", "rb").read()
        open(f"{sdir}/{victim}", "wb").write(data[: len(data) - 20000])
        c6 = dict(cfg, test_dicom_root=root, cache_dir=f"{W}/cache6_{case}", out_path=f"{W}/sub6_{case}/submission.csv")
        os.makedirs(f"{W}/sub6_{case}")
        sub6, rep6 = predict.run(c6)
        stages = sorted({(f["stage"], f["SeriesInstanceUID"] == big.SeriesInstanceUID) for f in rep6["decode_failures"]})
        cached = f"{W}/cache6_{case}/v2/{big.SeriesInstanceUID}.npy"
        T = np.load(cached, mmap_mode="r").shape[0] if os.path.exists(cached) else None
        if case == "end":
            assert ("dropped_end_slice", True) in stages and T == big.n - 1, (stages, T, big.n)
        else:
            assert ("decode", True) in stages and T is None, (stages, T)
        print(f"TEST 6 ({case}) ok: records {stages} | cached slices {T} of {big.n}")

    # 4) missing decoders must abort
    import pydicom.pixels as pp
    orig = pp.get_decoder
    class _Fake:
        is_available = False
    pp.get_decoder = lambda uid: _Fake()
    try:
        predict.assert_decoders()
        print("TEST 4 FAILED: no exception")
    except RuntimeError as ex:
        print("TEST 4 ok: aborted ->", str(ex)[:90])
    finally:
        pp.get_decoder = orig


if __name__ == "__main__":
    main()
