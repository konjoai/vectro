"""tests/test_v3_api.py — Phase 9: Unified v3 API (PQCodebook, HNSWIndex, VectroV3)."""

import numpy as np
import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.v3_api import (  # noqa: E402
    PQCodebook,
    HNSWIndex,
    V3Result,
    VectroV3,
    _is_cloud_uri,
    _is_vqz,
    _uri_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 96  # small dim so tests run quickly; divisible by sub-spaces 1..96
N_TRAIN = 128
N_DB = 32

rng = np.random.default_rng(42)
TRAIN_VECS = rng.standard_normal((N_TRAIN, D)).astype(np.float32)
DB_VECS = rng.standard_normal((N_DB, D)).astype(np.float32)
QUERY_VEC = rng.standard_normal(D).astype(np.float32)


@pytest.fixture(scope="module")
def pq_codebook():
    return PQCodebook.train(TRAIN_VECS, n_subspaces=8, n_centroids=32)


@pytest.fixture(scope="module")
def trained_rq():
    from python.rq_api import ResidualQuantizer

    rq = ResidualQuantizer(n_subspaces=8, n_passes=2)
    rq.train(TRAIN_VECS)
    return rq


# ---------------------------------------------------------------------------
# 1. PQCodebook
# ---------------------------------------------------------------------------


class TestPQCodebook:
    def test_train_returns_instance(self):
        cb = PQCodebook.train(TRAIN_VECS, n_subspaces=8, n_centroids=32)
        assert isinstance(cb, PQCodebook)

    def test_attributes(self, pq_codebook):
        assert pq_codebook.n_subspaces == 8
        assert pq_codebook.n_centroids == 32
        assert pq_codebook.sub_dim == D // 8

    def test_encode_shape(self, pq_codebook):
        codes = pq_codebook.encode(DB_VECS)
        assert codes.shape == (N_DB, 8)
        assert codes.dtype == np.uint8

    def test_decode_shape(self, pq_codebook):
        codes = pq_codebook.encode(DB_VECS)
        recon = pq_codebook.decode(codes)
        assert recon.shape == (N_DB, D)
        assert recon.dtype == np.float32

    def test_encode_decode_quality(self, pq_codebook):
        codes = pq_codebook.encode(TRAIN_VECS)
        recon = pq_codebook.decode(codes)
        norms_orig = np.linalg.norm(TRAIN_VECS, axis=1, keepdims=True)
        norms_recon = np.linalg.norm(recon, axis=1, keepdims=True)
        cos = np.sum(
            TRAIN_VECS / np.maximum(norms_orig, 1e-8) * recon / np.maximum(norms_recon, 1e-8),
            axis=1,
        ).mean()
        assert cos > 0.5  # coarse codebook (32 centroids), still reasonable

    def test_save_load_round_trip(self, pq_codebook, tmp_path):
        path = str(tmp_path / "cb.pqcb.npz")
        pq_codebook.save(path)
        cb2 = PQCodebook.load(path)
        assert cb2.n_subspaces == pq_codebook.n_subspaces
        np.testing.assert_array_almost_equal(cb2._cb.centroids, pq_codebook._cb.centroids)

    def test_compression_ratio(self, pq_codebook):
        ratio = pq_codebook.compression_ratio(D)
        assert ratio == pytest.approx(D * 4 / pq_codebook.n_subspaces)


# ---------------------------------------------------------------------------
# 2. HNSWIndex
# ---------------------------------------------------------------------------


class TestHNSWIndex:
    def test_add_batch_auto_ids(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        idx.add_batch(DB_VECS)
        assert len(idx) == N_DB

    def test_add_batch_user_ids(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        ids = [f"doc_{i}" for i in range(N_DB)]
        idx.add_batch(DB_VECS, ids=ids)
        assert idx._user_ids[0] == "doc_0"

    def test_add_batch_wrong_id_count(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        with pytest.raises(ValueError, match="ids length"):
            idx.add_batch(DB_VECS, ids=["only_one"])

    def test_search_returns_top_k(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        idx.add_batch(DB_VECS)
        ids, dists = idx.search(QUERY_VEC, top_k=5)
        assert len(ids) <= 5
        assert len(dists) <= 5

    def test_search_distances_non_negative(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        idx.add_batch(DB_VECS)
        ids, dists = idx.search(QUERY_VEC, top_k=5)
        assert (dists >= 0).all()

    def test_save_load_round_trip(self, tmp_path):
        idx = HNSWIndex(dim=D, M=4, ef_build=20, quantization="int8")
        ids = [f"v{i}" for i in range(N_DB)]
        idx.add_batch(DB_VECS, ids=ids)

        path = str(tmp_path / "test.hnsw")
        idx.save(path)
        idx2 = HNSWIndex.load(path)

        assert len(idx2) == len(idx)
        assert idx2._user_ids == idx._user_ids
        assert idx2._quantization == "int8"
        assert idx2._dim == D

    def test_empty_search_returns_empty(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        ids, dists = idx.search(QUERY_VEC, top_k=5)
        assert len(ids) == 0

    def test_add_single_vector(self):
        idx = HNSWIndex(dim=D, M=4, ef_build=20)
        idx.add_batch(QUERY_VEC)  # 1-D input
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# 3. V3Result
# ---------------------------------------------------------------------------


class TestV3Result:
    def test_default_data_empty(self):
        r = V3Result(profile="int8", n_vectors=10, dims=D)
        assert r.data == {}

    def test_fields(self):
        r = V3Result(profile="nf4", n_vectors=5, dims=128, data={"x": 1})
        assert r.profile == "nf4"
        assert r.n_vectors == 5
        assert r.dims == 128


# ---------------------------------------------------------------------------
# 4. VectroV3 — individual profiles
# ---------------------------------------------------------------------------


class TestVectroV3Int8:
    def test_compress_shape(self):
        v = VectroV3(profile="int8")
        result = v.compress(DB_VECS)
        assert isinstance(result, V3Result)
        assert result.profile == "int8"
        assert result.n_vectors == N_DB
        assert result.dims == D
        assert result.data["quantized"].shape == (N_DB, D)
        assert result.data["scales"].dtype == np.float32

    def test_decompress_shape(self):
        v = VectroV3(profile="int8")
        result = v.compress(DB_VECS)
        recon = v.decompress(result)
        assert recon.shape == (N_DB, D)
        assert recon.dtype == np.float32

    def test_single_vector(self):
        v = VectroV3(profile="int8")
        result = v.compress(DB_VECS[0])  # 1-D input
        assert result.n_vectors == 1


class TestVectroV3NF4:
    def test_compress_shape(self):
        v = VectroV3(profile="nf4")
        result = v.compress(DB_VECS)
        assert result.profile == "nf4"
        assert result.data["packed"].shape[1] == D // 2  # 2 nibbles per byte
        assert result.data["scales"].dtype == np.float32

    def test_decompress_shape(self):
        v = VectroV3(profile="nf4")
        result = v.compress(DB_VECS)
        recon = v.decompress(result)
        assert recon.shape == (N_DB, D)


class TestVectroV3PQ:
    def test_pq96_compress(self, pq_codebook):
        v = VectroV3(profile="pq-96", codebook=pq_codebook)
        # Use codebook with 8 sub-spaces (our fixture uses n_subspaces=8)
        v2 = VectroV3(profile="pq-48")
        # "pq-48" without codebook
        with pytest.raises(ValueError, match="codebook"):
            v2.compress(DB_VECS)

    def test_pq_compress_decode(self, pq_codebook):
        v = VectroV3(profile="pq-96", codebook=pq_codebook)
        result = v.compress(DB_VECS)
        assert result.data["codes"].shape == (N_DB, 8)
        recon = v.decompress(result)
        assert recon.shape == (N_DB, D)

    def test_pq_decompress_without_codebook(self, pq_codebook):
        v_enc = VectroV3(profile="pq-96", codebook=pq_codebook)
        result = v_enc.compress(DB_VECS)
        v_no_cb = VectroV3(profile="pq-96")
        with pytest.raises(ValueError, match="codebook"):
            v_no_cb.decompress(result)


class TestVectroV3Binary:
    def test_compress_shape(self):
        v = VectroV3(profile="binary")
        result = v.compress(DB_VECS)
        import math

        expected_packed_d = math.ceil(D / 8)
        assert result.data["packed"].shape == (N_DB, expected_packed_d)

    def test_decompress_shape(self):
        v = VectroV3(profile="binary")
        result = v.compress(DB_VECS)
        recon = v.decompress(result)
        assert recon.shape == (N_DB, D)


class TestVectroV3RQ:
    def test_compress_without_rq_raises(self):
        v = VectroV3(profile="rq-3pass")
        with pytest.raises(ValueError, match="ResidualQuantizer"):
            v.compress(DB_VECS)

    def test_train_rq_and_compress(self):
        v = VectroV3(profile="rq-3pass")
        v.train_rq(TRAIN_VECS, n_subspaces=8, n_passes=2)
        result = v.compress(DB_VECS)
        assert result.profile == "rq-3pass"
        assert isinstance(result.data["codes"], list)
        assert result.data["n_passes"] == 2

    def test_train_rq_and_decompress(self, trained_rq):
        v = VectroV3(profile="rq-3pass", rq=trained_rq)
        result = v.compress(DB_VECS)
        recon = v.decompress(result)
        assert recon.shape == (N_DB, D)

    def test_decompress_without_rq_raises(self, trained_rq):
        v_enc = VectroV3(profile="rq-3pass", rq=trained_rq)
        result = v_enc.compress(DB_VECS)
        v_no_rq = VectroV3(profile="rq-3pass")
        with pytest.raises(ValueError):
            v_no_rq.decompress(result)


class TestVectroV3InvalidProfile:
    def test_bad_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            VectroV3(profile="turbo-ultra")


# ---------------------------------------------------------------------------
# 5. auto_compress
# ---------------------------------------------------------------------------


class TestAutoCompress:
    def test_returns_dict(self):
        result = VectroV3.auto_compress(DB_VECS, target_cosine=0.70, target_compression=2.0)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = VectroV3.auto_compress(DB_VECS, target_cosine=0.70, target_compression=2.0)
        for key in ("mode", "cosine_sim", "compression_ratio"):
            assert key in result, f"Missing key: {key}"

    def test_cosine_sim_range(self):
        result = VectroV3.auto_compress(DB_VECS, target_cosine=0.70, target_compression=2.0)
        assert 0.0 <= result["cosine_sim"] <= 1.0


# ---------------------------------------------------------------------------
# 6. save_compressed / load_compressed — INT8
# ---------------------------------------------------------------------------


class TestSaveLoadInt8:
    def test_int8_local_round_trip(self, tmp_path):
        v = VectroV3(profile="int8")
        result = v.compress(DB_VECS)
        path = str(tmp_path / "int8.vqz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.profile == "int8"
        np.testing.assert_array_equal(loaded.data["quantized"], result.data["quantized"])
        np.testing.assert_array_almost_equal(loaded.data["scales"], result.data["scales"])

    def test_int8_dims_and_n_vectors_preserved(self, tmp_path):
        v = VectroV3(profile="int8")
        result = v.compress(DB_VECS)
        path = str(tmp_path / "meta.vqz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.dims == D
        assert loaded.n_vectors == N_DB


# ---------------------------------------------------------------------------
# 7. save_compressed / load_compressed — NF4
# ---------------------------------------------------------------------------


class TestSaveLoadNF4:
    def test_nf4_local_round_trip(self, tmp_path):
        v = VectroV3(profile="nf4")
        result = v.compress(DB_VECS)
        path = str(tmp_path / "nf4.vqz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.profile == "nf4"
        np.testing.assert_array_equal(loaded.data["packed"], result.data["packed"])
        np.testing.assert_array_almost_equal(loaded.data["scales"], result.data["scales"])


# ---------------------------------------------------------------------------
# 8. save_compressed / load_compressed — Binary
# ---------------------------------------------------------------------------


class TestSaveLoadBinary:
    def test_binary_local_round_trip(self, tmp_path):
        v = VectroV3(profile="binary")
        result = v.compress(DB_VECS)
        path = str(tmp_path / "binary.npz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.profile == "binary"
        np.testing.assert_array_equal(loaded.data["packed"], result.data["packed"])


# ---------------------------------------------------------------------------
# 9. save_compressed / load_compressed — PQ
# ---------------------------------------------------------------------------


class TestSaveLoadPQ:
    def test_pq_local_round_trip(self, pq_codebook, tmp_path):
        v = VectroV3(profile="pq-96", codebook=pq_codebook)
        result = v.compress(DB_VECS)
        path = str(tmp_path / "pq.npz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.profile == "pq-96"
        np.testing.assert_array_equal(loaded.data["codes"], result.data["codes"])


# ---------------------------------------------------------------------------
# 10. save_compressed / load_compressed — RQ
# ---------------------------------------------------------------------------


class TestSaveLoadRQ:
    def test_rq_local_round_trip(self, trained_rq, tmp_path):
        v = VectroV3(profile="rq-3pass", rq=trained_rq)
        result = v.compress(DB_VECS)
        path = str(tmp_path / "rq.npz")
        v.save_compressed(result, path)
        loaded = v.load_compressed(path)
        assert loaded.profile == "rq-3pass"
        assert isinstance(loaded.data["codes"], list)
        assert len(loaded.data["codes"]) == result.data["n_passes"]


# ---------------------------------------------------------------------------
# 11. Cloud URI helpers (unit tests; no real I/O)
# ---------------------------------------------------------------------------


class TestCloudHelpers:
    def test_is_cloud_uri_s3(self):
        assert _is_cloud_uri("s3://my-bucket/key.vqz") is True

    def test_is_cloud_uri_gs(self):
        assert _is_cloud_uri("gs://my-bucket/key.vqz") is True

    def test_is_cloud_uri_abfs(self):
        assert _is_cloud_uri("abfs://container/key.vqz") is True

    def test_is_cloud_uri_local(self):
        assert _is_cloud_uri("/tmp/file.vqz") is False
        assert _is_cloud_uri("relative/path.vqz") is False

    def test_uri_key_s3(self):
        assert _uri_key("s3://bucket/prefix/file.vqz") == "prefix/file.vqz"

    def test_uri_key_no_prefix(self):
        assert _uri_key("s3://bucket/file.vqz") == "file.vqz"

    def test_is_vqz_detects_magic(self, tmp_path):
        path = str(tmp_path / "ok.vqz")
        with open(path, "wb") as fh:
            fh.write(b"VECTRO\x03\x00" + b"\x00" * 56)
        assert _is_vqz(path) is True

    def test_is_vqz_rejects_npz(self, tmp_path):
        path = str(tmp_path / "not.vqz")
        with open(path, "wb") as fh:
            fh.write(b"PK\x03\x04" + b"\x00" * 60)
        assert _is_vqz(path) is False

    def test_cloud_save_non_int8_raises_without_fsspec(self, tmp_path):
        v = VectroV3(profile="binary")
        result = v.compress(DB_VECS)
        with pytest.raises(NotImplementedError, match="cloud save"):
            v.save_compressed(result, "s3://bucket/out.vqz")
