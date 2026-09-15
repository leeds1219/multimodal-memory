# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LIBERO-Plus and LIBERO-PRO benchmark support."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from vlm_orchestrator.aux_benchmarks.libero_specs import (
    PLUS_CATEGORIES,
    PLUS_SUITES,
    PRO_BASE_SUITES,
    PRO_DIM_TO_SUFFIX,
    PlusClassification,
    PlusResults,
    PlusTask,
    ProPerturbConfig,
    ProResults,
    aggregate_plus_results,
    aggregate_pro_results,
    format_plus_report,
    format_pro_report,
    get_all_pro_suites,
    get_experiment_plan,
    get_pro_suite_names,
    infer_pro_dimension_from_suite,
    load_plus_classification,
    load_pro_config,
    make_benchmark_spec,
)


# ═══════════════════════════════════════════════════════════════════
#  LIBERO-Plus Tests
# ═══════════════════════════════════════════════════════════════════


class TestPlusClassification:
    """Tests for LIBERO-Plus task classification parsing."""

    @pytest.fixture
    def sample_classification_json(self, tmp_path: Path) -> Path:
        """Create a minimal task_classification.json for testing."""
        data = {
            "libero_spatial": [
                {"id": 1, "name": "task_a_table_1", "category": "Background Textures", "difficulty_level": 2},
                {"id": 2, "name": "task_a_view_0_0", "category": "Camera Viewpoints", "difficulty_level": 3},
                {"id": 3, "name": "task_b_language_1", "category": "Language Instructions", "difficulty_level": 1},
                {"id": 4, "name": "task_c_light_1", "category": "Light Conditions", "difficulty_level": 4},
            ],
            "libero_goal": [
                {"id": 5, "name": "task_d_add_1", "category": "Objects Layout", "difficulty_level": 2},
                {"id": 6, "name": "task_e_initstate_1", "category": "Robot Initial States", "difficulty_level": 1},
                {"id": 7, "name": "task_f_noise_1", "category": "Sensor Noise", "difficulty_level": 5},
            ],
        }
        # Create expected directory structure
        benchmark_dir = tmp_path / "libero" / "libero" / "benchmark"
        benchmark_dir.mkdir(parents=True)
        json_path = benchmark_dir / "task_classification.json"
        json_path.write_text(json.dumps(data))
        return tmp_path

    def test_load_classification(self, sample_classification_json: Path):
        cls = load_plus_classification(sample_classification_json)
        assert cls._loaded
        assert cls.total_count == 7
        assert len(cls.tasks_by_suite) == 2
        assert len(cls.tasks_by_suite["libero_spatial"]) == 4
        assert len(cls.tasks_by_suite["libero_goal"]) == 3

    def test_tasks_by_category(self, sample_classification_json: Path):
        cls = load_plus_classification(sample_classification_json)
        bg_tasks = cls.tasks_by_category("Background Textures")
        assert len(bg_tasks) == 1
        assert bg_tasks[0].name == "task_a_table_1"

    def test_tasks_by_suite_and_category(self, sample_classification_json: Path):
        cls = load_plus_classification(sample_classification_json)
        tasks = cls.tasks_by_suite_and_category("libero_spatial", "Camera Viewpoints")
        assert len(tasks) == 1
        assert tasks[0].difficulty_level == 3

    def test_category_counts(self, sample_classification_json: Path):
        cls = load_plus_classification(sample_classification_json)
        counts = cls.category_counts()
        assert counts["Background Textures"] == 1
        assert counts["Camera Viewpoints"] == 1
        assert counts["Objects Layout"] == 1

    def test_difficulty_distribution(self, sample_classification_json: Path):
        cls = load_plus_classification(sample_classification_json)
        dist = cls.difficulty_distribution()
        assert dist[1] == 2  # difficulty 1: language + robot init
        assert dist[2] == 2  # difficulty 2: bg textures + objects layout

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="task_classification.json"):
            load_plus_classification(tmp_path)


class TestPlusResultsAggregation:
    """Tests for LIBERO-Plus result aggregation."""

    def test_aggregate_basic(self):
        cls = PlusClassification()
        cls.tasks_by_suite["libero_spatial"] = [
            PlusTask(id=1, name="task_a", category="Background Textures", difficulty_level=2, suite="libero_spatial"),
            PlusTask(id=2, name="task_b", category="Camera Viewpoints", difficulty_level=3, suite="libero_spatial"),
            PlusTask(id=3, name="task_c", category="Language Instructions", difficulty_level=1, suite="libero_spatial"),
        ]

        episodes = [
            {"task_name": "task_a", "success": True},
            {"task_name": "task_b", "success": False},
            {"task_name": "task_c", "success": True},
        ]

        results = aggregate_plus_results(episodes, cls)
        assert results.total_episodes == 3
        assert results.total_successes == 2
        assert results.overall_rate == pytest.approx(66.67, abs=0.1)

        assert results.category_results["Background Textures"]["rate"] == 100.0
        assert results.category_results["Camera Viewpoints"]["rate"] == 0.0
        assert results.category_results["Language Instructions"]["rate"] == 100.0

    def test_aggregate_unknown_tasks_ignored(self):
        cls = PlusClassification()
        cls.tasks_by_suite["libero_spatial"] = [
            PlusTask(id=1, name="known_task", category="Sensor Noise", difficulty_level=1, suite="libero_spatial"),
        ]

        episodes = [
            {"task_name": "known_task", "success": True},
            {"task_name": "unknown_task", "success": False},
        ]

        results = aggregate_plus_results(episodes, cls)
        assert results.total_episodes == 2
        assert results.total_successes == 1
        # category_results only has entries for known tasks
        assert "Sensor Noise" in results.category_results
        assert results.category_results["Sensor Noise"]["episodes"] == 1

    def test_format_report(self):
        results = PlusResults(
            total_episodes=100,
            total_successes=53,
            category_results={
                "Background Textures": {"episodes": 20, "successes": 15, "rate": 75.0},
                "Camera Viewpoints": {"episodes": 20, "successes": 5, "rate": 25.0},
            },
            suite_results={
                "libero_spatial": {"episodes": 50, "successes": 30, "rate": 60.0},
            },
        )
        report = format_plus_report(results)
        assert "LIBERO-Plus" in report
        assert "53/100" in report
        assert "53.0%" in report
        assert "Background" in report


# ═══════════════════════════════════════════════════════════════════
#  LIBERO-PRO Tests
# ═══════════════════════════════════════════════════════════════════


class TestProConfig:
    """Tests for LIBERO-PRO configuration loading."""

    @pytest.fixture
    def sample_pro_config(self, tmp_path: Path) -> Path:
        """Create a minimal evaluation_config.yaml."""
        config = {
            "bddl_files_path": "./LIBERO-PRO/libero/libero/bddl_files/",
            "script_path": "./LIBERO-PRO/notebooks/generate_init_states.py",
            "init_file_dir": "./LIBERO-PRO/libero/libero/init_files/",
            "use_environment": True,
            "use_swap": False,
            "use_object": True,
            "use_language": False,
            "use_task": False,
            "ood_task_configs": {
                "environment": "./libero_ood/ood_environment.yaml",
                "swap": "./libero_ood/ood_spatial_relation.yaml",
                "object": "./libero_ood/ood_object.yaml",
                "language": "./libero_ood/ood_language.yaml",
                "task": "./libero_ood/ood_task.yaml",
            },
            "perturbation_mapping": {
                "use_environment": "env",
                "use_swap": "swap",
                "use_object": "object",
                "use_language": "lan",
                "use_task": "task",
            },
        }
        import yaml
        config_path = tmp_path / "evaluation_config.yaml"
        config_path.write_text(yaml.dump(config))
        return tmp_path

    def test_load_config(self, sample_pro_config: Path):
        config = load_pro_config(root=sample_pro_config)
        assert config.use_environment is True
        assert config.use_swap is False
        assert config.use_object is True
        assert config.use_language is False
        assert config.use_task is False

    def test_active_dimensions(self, sample_pro_config: Path):
        config = load_pro_config(root=sample_pro_config)
        dims = config.active_dimensions
        assert "object" in dims
        assert "environment" in dims
        assert "position" not in dims  # use_swap is False
        assert "semantic" not in dims

    def test_config_tag(self, sample_pro_config: Path):
        config = load_pro_config(root=sample_pro_config)
        tag = config.config_tag
        assert "object" in tag
        assert "environment" in tag

    def test_config_tag_empty(self):
        config = ProPerturbConfig()
        assert config.config_tag == "original"


class TestProSuiteNames:
    """Tests for LIBERO-PRO suite name utilities."""

    def test_get_suite_names_single_dim(self):
        names = get_pro_suite_names("libero_goal", "position")
        assert names == ["libero_goal_swap"]

    def test_get_suite_names_all_dims(self):
        names = get_pro_suite_names("libero_goal")
        assert len(names) == 5
        assert "libero_goal_object" in names
        assert "libero_goal_swap" in names
        assert "libero_goal_lan" in names
        assert "libero_goal_task" in names
        assert "libero_goal_env" in names

    def test_get_all_pro_suites(self):
        all_suites = get_all_pro_suites()
        assert len(all_suites) == 20  # 4 bases × 5 dims

    def test_infer_dimension(self):
        assert infer_pro_dimension_from_suite("libero_goal_swap") == "position"
        assert infer_pro_dimension_from_suite("libero_10_lan") == "semantic"
        assert infer_pro_dimension_from_suite("libero_object_env") == "environment"
        assert infer_pro_dimension_from_suite("libero_spatial_task") == "task"
        assert infer_pro_dimension_from_suite("libero_goal_object") == "object"

    def test_infer_dimension_unknown(self):
        assert infer_pro_dimension_from_suite("libero_goal") is None
        assert infer_pro_dimension_from_suite("random_suite") is None

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="Unknown PRO dimension"):
            get_pro_suite_names("libero_goal", "nonexistent")


class TestProResultsAggregation:
    """Tests for LIBERO-PRO result aggregation."""

    def test_aggregate_basic(self):
        suite_results = {
            "libero_goal": {
                "total_episodes": 500, "total_successes": 485, "success_rate": 97.0,
            },
            "libero_goal_swap": {
                "total_episodes": 500, "total_successes": 0, "success_rate": 0.0,
            },
            "libero_goal_lan": {
                "total_episodes": 500, "total_successes": 350, "success_rate": 70.0,
            },
            "libero_goal_object": {
                "total_episodes": 500, "total_successes": 200, "success_rate": 40.0,
            },
        }

        results = aggregate_pro_results(suite_results)
        assert results.total_episodes == 2000
        assert results.total_successes == 1035

        # Base suite should be in base_results
        assert "libero_goal" in results.base_results
        assert results.base_results["libero_goal"]["rate"] == 97.0

        # Per-dimension
        assert results.dimension_results["position"]["rate"] == 0.0
        assert results.dimension_results["semantic"]["rate"] == 70.0
        assert results.dimension_results["object"]["rate"] == 40.0

    def test_format_report(self):
        results = ProResults(
            total_episodes=1000,
            total_successes=530,
            base_results={
                "libero_goal": {"episodes": 500, "successes": 485, "rate": 97.0},
            },
            dimension_results={
                "position": {"episodes": 100, "successes": 0, "rate": 0.0},
                "semantic": {"episodes": 100, "successes": 70, "rate": 70.0},
            },
        )
        report = format_pro_report(results)
        assert "LIBERO-PRO" in report
        assert "530/1000" in report
        assert "position" in report
        assert "0.0%" in report


# ═══════════════════════════════════════════════════════════════════
#  Unified Benchmark Tests
# ═══════════════════════════════════════════════════════════════════


class TestBenchmarkSpec:
    """Tests for unified benchmark specification."""

    def test_make_libero_spec(self):
        spec = make_benchmark_spec("libero")
        assert spec.name == "libero"
        assert len(spec.suites) == 4
        assert spec.num_trials_per_task == 50

    def test_make_plus_spec(self):
        spec = make_benchmark_spec("libero_plus")
        assert spec.name == "libero_plus"
        assert spec.num_trials_per_task == 1  # each task is unique
        assert len(spec.suites) == 4

    def test_make_pro_spec(self):
        spec = make_benchmark_spec("libero_pro")
        assert spec.name == "libero_pro"
        assert len(spec.suites) == 20  # 4 × 5

    def test_make_pro_spec_filtered(self):
        spec = make_benchmark_spec("libero_pro", pro_dimensions=["position"])
        assert len(spec.suites) == 4  # 4 base suites × 1 dim
        assert all("swap" in s for s in spec.suites)

    def test_make_spec_custom_suites(self):
        spec = make_benchmark_spec("libero", suites=["libero_10"])
        assert spec.suites == ["libero_10"]

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            make_benchmark_spec("nonexistent")


class TestExperimentPlan:
    """Tests for experiment plan generation."""

    def test_plus_plan(self):
        plan = get_experiment_plan("libero_plus")
        assert len(plan) > 0
        # Should have entries for each strategy × suite × category
        strategies = set(e["strategy"] for e in plan)
        assert "passthrough" in strategies
        assert "subgoal_scene_edit" in strategies

        # Check relevance flags
        lang_entries = [e for e in plan if e["category"] == "Language Instructions"]
        subgoal_lang = [e for e in lang_entries if e["strategy"] == "subgoal"]
        assert all(e["is_relevant"] for e in subgoal_lang)

        sensor_entries = [e for e in plan if e["category"] == "Sensor Noise"]
        assert all(not e["is_relevant"] for e in sensor_entries)

    def test_pro_plan(self):
        plan = get_experiment_plan("libero_pro")
        assert len(plan) > 0

        pos_entries = [e for e in plan if e["dimension"] == "position"]
        assert len(pos_entries) > 0
        # subgoal_scene_edit should be relevant for position
        sse_pos = [e for e in pos_entries if e["strategy"] == "subgoal_scene_edit"]
        assert all(e["is_relevant"] for e in sse_pos)


# ═══════════════════════════════════════════════════════════════════
#  Integration: Test with Real Files (if available)
# ═══════════════════════════════════════════════════════════════════


class TestRealPlusClassification:
    """Test loading real LIBERO-Plus classification (skipped if not available)."""

    @pytest.fixture
    def plus_root(self) -> Path:
        root = Path(os.path.expanduser("~/LIBERO-plus"))
        if not (root / "libero" / "libero" / "benchmark" / "task_classification.json").exists():
            pytest.skip("LIBERO-Plus repo not available")
        return root

    def test_load_real_classification(self, plus_root: Path):
        cls = load_plus_classification(plus_root)
        assert cls.total_count > 10000  # Should be ~10030
        assert len(cls.tasks_by_suite) == 4

        # Verify all 7 categories are present
        all_cats = set(t.category for t in cls.all_tasks)
        for cat in PLUS_CATEGORIES:
            assert cat in all_cats, f"Missing category: {cat}"

    def test_per_suite_counts(self, plus_root: Path):
        cls = load_plus_classification(plus_root)
        for suite in PLUS_SUITES:
            tasks = cls.tasks_by_suite.get(suite, [])
            assert len(tasks) > 2000, f"{suite} should have >2000 tasks, got {len(tasks)}"


class TestRealProConfig:
    """Test loading real LIBERO-PRO config (skipped if not available)."""

    @pytest.fixture
    def pro_root(self) -> Path:
        root = Path(os.path.expanduser("~/LIBERO-PRO"))
        if not (root / "evaluation_config.yaml").exists():
            pytest.skip("LIBERO-PRO repo not available")
        return root

    def test_load_real_config(self, pro_root: Path):
        config = load_pro_config(root=pro_root)
        assert config.bddl_files_path != ""
        assert len(config.ood_task_configs) >= 5


# ═══════════════════════════════════════════════════════════════════
#  Eval Client Max Steps Extension
# ═══════════════════════════════════════════════════════════════════


class TestMaxSteps:
    """Test that get_max_steps handles extended suite names.

    These tests import ``get_max_steps`` from the LIBERO eval client, which
    has heavy dependencies (imageio, openpi_client, etc.) that may not be
    available in the test environment.  We skip if import fails.
    """

    @pytest.fixture(autouse=True)
    def _import_guard(self):
        try:
            from examples.libero.libero_eval_client import get_max_steps
            self.get_max_steps = get_max_steps
        except ImportError as e:
            pytest.skip(f"Cannot import libero_eval_client: {e}")

    def test_base_suites(self):
        assert self.get_max_steps("libero_spatial") == 220
        assert self.get_max_steps("libero_goal") == 300
        assert self.get_max_steps("libero_10") == 520

    def test_pro_suites_inherit(self):
        assert self.get_max_steps("libero_goal_swap") == 300
        assert self.get_max_steps("libero_10_lan") == 520
        assert self.get_max_steps("libero_spatial_env") == 220
        assert self.get_max_steps("libero_object_task") == 280

    def test_unknown_suite_default(self):
        # Should return 400 default without raising
        assert self.get_max_steps("completely_unknown_suite") == 400


# ════════════════════════════════════════════════════════════════
# LIBERO-Mem tests
# ════════════════════════════════════════════════════════════════

class TestMemBenchmark:
    """Tests for LIBERO-Mem support."""

    def test_make_benchmark_spec_mem(self):
        from vlm_orchestrator.aux_benchmarks.libero_specs import make_benchmark_spec
        spec = make_benchmark_spec("libero_mem")
        assert spec.name == "libero_mem"
        assert spec.suites == ["libero_mem"]
        assert spec.num_trials_per_task == 20

    def test_make_benchmark_spec_mem_custom_trials(self):
        from vlm_orchestrator.aux_benchmarks.libero_specs import make_benchmark_spec
        spec = make_benchmark_spec("libero_mem", num_trials=5)
        assert spec.num_trials_per_task == 5

    def test_aggregate_mem_results(self):
        from vlm_orchestrator.aux_benchmarks.libero_specs import aggregate_mem_results
        results_json = {
            "total_episodes": 10,
            "total_successes": 2,
            "subgoal_completion_rate": 0.35,
            "total_subgoals_completed": 8,
            "total_subgoals_possible": 31,
            "per_task": [
                {
                    "task_id": 0,
                    "language": "pick up bowl",
                    "n_subgoals": 1,
                    "episodes": [
                        {"success": True, "tiered_success": 1.0},
                    ],
                },
                {
                    "task_id": 4,
                    "language": "bowl 5 times",
                    "n_subgoals": 5,
                    "episodes": [
                        {"success": False, "tiered_success": 0.6},
                    ],
                },
            ],
        }
        mem = aggregate_mem_results(results_json)
        assert mem.total_episodes == 10
        assert mem.total_successes == 2
        assert mem.binary_success_rate == 20.0
        assert len(mem.per_task) == 2
        assert mem.per_task[0]["binary_rate"] == 100.0
        assert mem.per_task[1]["subgoal_completion_rate"] == 0.6

    def test_aggregate_mem_tiers(self):
        from vlm_orchestrator.aux_benchmarks.libero_specs import aggregate_mem_results
        # Build results with all 10 tasks
        per_task = []
        for i in range(10):
            per_task.append({
                "task_id": i,
                "language": f"task {i+1}",
                "n_subgoals": [1,1,3,3,5,7,3,4,2,2][i],
                "episodes": [
                    {"success": i < 2, "tiered_success": 1.0 if i < 2 else 0.0},
                ],
            })
        results_json = {
            "total_episodes": 10,
            "total_successes": 2,
            "subgoal_completion_rate": 0.2,
            "total_subgoals_completed": 2,
            "total_subgoals_possible": 31,
            "per_task": per_task,
        }
        mem = aggregate_mem_results(results_json)
        # Easy tier = T1, T2 (task_ids 0, 1) — both succeeded
        assert mem.tier_results["easy"]["successes"] == 2
        assert mem.tier_results["easy"]["episodes"] == 2
        # Hard tier = T5-T8 (task_ids 4,5,6,7) — all failed
        assert mem.tier_results["hard"]["successes"] == 0

    def test_format_mem_results(self):
        from vlm_orchestrator.aux_benchmarks.libero_specs import (
            aggregate_mem_results, format_mem_results,
        )
        results_json = {
            "total_episodes": 2,
            "total_successes": 1,
            "subgoal_completion_rate": 0.5,
            "total_subgoals_completed": 3,
            "total_subgoals_possible": 6,
            "per_task": [
                {
                    "task_id": 0,
                    "language": "pick up bowl",
                    "n_subgoals": 1,
                    "episodes": [{"success": True, "tiered_success": 1.0}],
                },
                {
                    "task_id": 6,
                    "language": "swap bowls",
                    "n_subgoals": 3,
                    "episodes": [{"success": False, "tiered_success": 0.33}],
                },
            ],
        }
        mem = aggregate_mem_results(results_json)
        report = format_mem_results(mem)
        assert "LIBERO-Mem" in report
        assert "50.0%" in report
        assert "Per-Task" in report
        assert "Tier Summary" in report
