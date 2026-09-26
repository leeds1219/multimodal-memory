# Reported success-rate numbers from baseline papers

Machine-readable version: `analysis/paper_numbers.json`. This file explains sources, table
numbers, and caveats. All numbers were extracted from live arXiv HTML fetches (WebFetch) by
parallel research agents; **no number below or in the JSON was invented** -- where a table could
not be retrieved (image, truncation, size limit), that is stated explicitly instead.

Our task list: `configs/tasks.yaml` (70 tasks, uids `<group>_<local_id>`), targets in
`configs/task_targets.yaml`. Groups: wooden(11), stone(10), iron(16), gold(7), redstone(6),
diamond(7), armor(13) = 70.

---

## 1. MineEvolve -- arXiv 2603.13131v3

**Source used:** `https://arxiv.org/html/2603.13131v3` (fetched and cross-verified by parsing the
raw HTML `<table>` DOM directly, matching a separate WebFetch summarization pass byte-for-byte).

**Title:** "MineEvolve: Self-Evolution with Accumulated Knowledge for Long-Horizon Embodied
Minecraft Agents." Code: `github.com/xzw-ustc/MC-MineEvolve`.

**Setting:** 70-task benchmark drawn from the "MCU" tech-tree suite (Zheng et al. 2025, ICML).
Group counts (Wooden 11, Stone 10, Iron 16, Gold 7, Redstone 6, Diamond 7, Armor 13 = 70) and
group descriptions (Table 3, Appendix D.1) line up with our 70 tasks group-for-group. 5 planner
backbones evaluated: Qwen3.5-Flash, Qwen3.5-Plus, GLM-4.7, **Gemini-3-Flash**, GPT-5.5, each
crossed with methods DEPS / JARVIS-1 / Optimus-1 / Optimus-2 (own GOAP controller) / MineEvolve
(all four LLM-planner methods share STEVE-1 as the low-level controller). A STEVE-1-only
(no-planner) row is also given. **Runs-per-task and horizon are asserted "held constant across
methods" but no concrete numbers are published anywhere in the retrieved sections.**

**Table 4** ("Full results on the MCU task suite"): full 5-planner x 5-method grid of per-group SR
(%) plus Overall (task-weighted over 70 tasks). Reproduced in full in the JSON under
`mineevolve.groups["<method>|<planner>"]`. **Table 1** additionally gives the same Overall plus
Weighted Easy (Wooden+Stone+Gold, 28 tasks) and Weighted Hard (Iron+Redstone+Diamond+Armor, 42
tasks) averages -- folded into each group entry as `easy_avg` / `hard_avg`.

**Appendix D.2 "Full Task-Group Results":** checked directly -- contains only Table 4 (group-level),
**no per-task numbers exist anywhere in the paper** (all 10 tables checked). Do not expect
uid-level ground truth from this paper; only group-level / Easy-Hard-Overall comparison is
possible, and even that is "approx" at the uid level since the paper never lists individual task
names.

**Caveat:** arXiv ID 2603.13131 = a March-2026-dated submission (v3 dated 10 May 2026) -- unusual
relative to the assistant's Jan-2026 knowledge cutoff but plausible given "today" is 2026-09-26;
fetch succeeded and content was internally consistent.

---

## 2. Optimus-1 -- arXiv 2408.03615

**Source used:** `https://arxiv.org/html/2408.03615` (latest = v2, NeurIPS 2024 camera-ready);
Appendix F.1 cross-checked via `https://ar5iv.labs.arxiv.org/html/2408.03615`.

**Setting:** Planner = GPT-4V (Knowledge-Guided Planner + Experience-Driven Reflector); Controller
= STEVE-1; Environment = MineRL, Minecraft 1.16.5, 20 FPS; empty starting inventory; memory =
Hierarchical Directed Knowledge Graph + Abstracted Multimodal Experience Pool. Runs per task: "at
least 30" per the stated protocol, but Appendix F.1's actual "Eval Times" column ranges 30-70 (not
a fixed 30 -- likely because simple items are reused as sub-goals inside other tasks' runs).
**Paper's own benchmark has 67 tasks (not 70)** -- confirmed mismatch for Wood (paper 9 vs our 11)
and Stone (paper 8 vs our 10) from the retrieved appendix tables.

**Table 1** ("Main Results"): per-group SR/AT(s)/AS for GPT-3.5, GPT-4V, DEPS, JARVIS-1,
Optimus-1, Human-level, across Wood/Stone/Iron/Gold/Diamond/Redstone/Armor. **Important:** the
"Overall" column is the mean of Iron+Gold+Diamond+Redstone+Armor **only** (verified by arithmetic
check against Optimus-1's own row: (46.69+8.51+11.61+25.02+19.47)/5 = 22.26, matching the reported
Overall) -- Wood and Stone are excluded from Overall in this paper. Full data (including AT/AS,
with `null` standing in for the paper's reported "+inf") is in `optimus1.groups`.

**Appendix F.1 (Tables 8-14), per-task:** only reports **Optimus-1's own** per-task numbers (no
baseline breakdown per task). Only **Table 8 (Wood, 9/9 tasks)** and **Table 9 (Stone, 8/8 tasks,
last cell's AT truncated)** were retrievable -- these map cleanly onto our wooden_00-06 and
stone_00-05 uids (all "exact"). Tables for Iron/Gold/Diamond/Redstone/Armor per-task breakdowns
exist in the paper (referenced via Table 1's "see Appendix F") but could **not** be fetched: 4
separate WebFetch attempts (different phrasings, both arxiv.org and ar5iv mirrors) all truncated at
the identical point mid-Table-9; the PDF fetch also failed outright (file >10 MB, tool hard cap).
These numbers are genuinely missing from this research pass, not fabricated.

---

## 3. JARVIS-1 -- arXiv 2311.05997

**Source used:** `https://ar5iv.labs.arxiv.org/html/2311.05997` (got furthest into the appendix
before truncation), cross-checked against `https://arxiv.org/html/2311.05997`.

**Setting corrections vs. the task brief:** JARVIS-1 explicitly does **NOT** use MineDojo's
interface -- the paper contrasts itself against MineDojo/GITM/Voyager and uses vanilla Minecraft's
native human interface (keyboard/mouse, 20 FPS). Table 1 confirms the brief's premise about
initial inventory: **Wood and Wood-Variants groups start empty; every other group (Stone, Iron,
Gold, Diamond, Redstone, Blocks, Armor, Decoration, Food) starts with an `iron_axe`** already in
inventory. Runs per task: "at least 30" stated, but actual per-task Eval Times in the appendix
range 30-86. Horizon: 12k steps (~10 min at 20 FPS) for most groups, 36k steps (~30 min) for
Gold/Diamond/Redstone. Benchmark = "Universal Benchmark," 200+ tasks across 11 groups (Wood 34,
Wood-Variants 43, Stone 10, Iron 22, Gold 9, Diamond 7, Redstone 7, Blocks 15, Armor 17,
Decoration 17, Food 9).

**Table 2** ("Results of JARVIS-1 and baselines"): group-level SR (mean +/- std) for baselines
**GPT, ReAct, Inner Monologue, DEPS, JARVIS-1** (note: this table does *not* include
VPT/Plan4MC/MineAgent, contrary to what the task brief assumed -- those appear only narratively in
Sec 4.3's single "diamond pickaxe" long-horizon comparison, where JARVIS-1 gets 6.22% vs VPT's
2.5% within ~20 min, improving to 12.5% with an extended time budget). No "Overall" row was
captured.

**Per-task appendix tables (Tables 5-8), JARVIS-1 only (no baseline per-task breakdown):**
- **Table 6 (Stone), complete, 10/10 tasks** -- all 7 of our craft-type stone uids match exactly.
- **Table 7 (Iron), complete, 22/22 tasks** -- 12 of our 16 iron uids match exactly (iron_13/14/15
  have no equivalent single-mine-count task; iron_ingot bundles mine+smelt).
- **Table 5 (Wood), partial, 9/34 tasks** -- our 7 craft-type wooden uids match exactly;
  wooden_07/08/09/10 not in the retrieved rows.
- **Table 8 (Gold), partial, 3/9 tasks** -- only golden_pickaxe, golden_sword match (gold_00,
  gold_02); gold_06 (ingot) comes from a Table 2 representative row instead (14.49%).
- **Tables 9-15 (Diamond, Redstone, Blocks, Armor, Decoration, Food full tables): NOT retrievable.**
  Every fetch (multiple mirrors/versions/prompts) truncated at the identical point mid-Table-8 --
  a fixed content-size cutoff, not an image/figure problem (everything reached is real HTML text).
  A handful of individual numbers for these groups (diamond, diamond_pickaxe, dropper,
  iron_helmet) come from Table 2's representative-row spot-checks, not the full appendix.
- Diamond/Redstone/Armor rows beyond those spot values are marked `"match": "approx"` with
  `"sr": null` in the JSON -- their existence in the relevant group is inferred, not confirmed with
  a number.

---

## 4. DEPS -- arXiv 2302.01560v3

**Source used:** `https://arxiv.org/html/2302.01560v3`, cross-checked against
`https://ar5iv.labs.arxiv.org/html/2302.01560` (identical content, including the same anomaly
noted below).

**Table numbering discrepancy:** the task brief expected "Table 3" for the MT1-MT8 results; in the
fetched v3 the main results table is actually **Table 2** ("Success rates of DEPS and existing LLM
planners on Minecraft Task101"). Table 3 in this version is a separate MineRL/MC-TextWorld
cross-version robustness check, not reproduced here.

**Setting corrections vs. the task brief:** Controller is **MC-Controller** (imitation learning),
**not STEVE-1** -- STEVE-1 appears only as a bibliography citation to later related work.
Environment = MineDojo, Minecraft 1.11.2. 30 runs/task. Horizon (max steps) per group: MT1 3000,
MT2 3000, MT3 6000, MT4 3000, MT5 6000, MT6 6000, MT7 6000, MT8 12000. Every task starts with an
axe given as a tool, otherwise empty inventory.

**MT1-MT8 definitions** (Table 1 counts + max-steps; Appendix B Tables 9-16 item lists) are in
`deps.setting.mt_group_definitions` in the JSON. Key overlaps with our groups: MT2 ("Tool Simple,"
12 tasks) spans **both** our wooden and stone tool-crafting tasks; MT6 ("Tool Complex") and MT7
("IronStage") together cover most of our iron group; MT8 is exactly our diamond_00 (mine diamond,
"ObtainDiamond"); MT5 ("Equipment") approximately covers our armor group (paper tests *equipping*
pre-existing armor, not necessarily crafting it). **No gold or redstone items appear anywhere in
MT1-MT8** -- DEPS's 2023-era MineDojo task set predates/excludes those tiers. One unresolved
internal inconsistency: Table 1 states MT4 has 6 tasks but the Appendix B item list attributed to
MT4 lists 13 items (reproduced identically on 2 independent fetches).

**Table 2 gives only MT-group averages, not per-task SR.** Appendix F ("Success Rates of ALL Tasks
in Minecraft") is referenced but its row data did not come through the fetch (only the caption did)
-- so no numeric per-task SR exists for the full DEPS pipeline in this research pass. A **separate**
low-level "imitation-learning controller" benchmark (**Table 18**) does give real per-skill numbers
(e.g., mine iron ore w/ stone pickaxe 40%, mine 3 iron ore 16%, mine diamond w/ iron pickaxe 35%,
kill cow 60%) -- these are included in the JSON under
`deps.tasks["DEPS-controller-only(Table18)|MC-Controller-IL"]`, clearly labeled as a **different
experiment** (atomic controller skill execution with no LLM planning loop, not the end-to-end DEPS
pipeline) so it should not be equated with full-pipeline task SR.

---

## 5. Optimus-3 -- arXiv 2506.10357v2

**Source used:** `https://arxiv.org/html/2506.10357v2` (v2, dated 2026-02-10 -- a substantial
revision of the June-2025 v1; compares against GPT-5-Instant/Thinking and Gemini-2.5, which
postdate v1). Full title: "Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent with
Dual-Granularity Reasoning-Aware Policy Optimization." Numbers parsed directly from the HTML
`<table>` DOM (not an LLM summarization pass) and cross-checked against the abstract's claimed
deltas -- high confidence.

**Setting:** Environment = MineRL (20 Hz), empty inventory, randomized spawn. >=30 rollouts/task.
**The 67-task Long-Horizon benchmark is inherited unchanged from the Optimus-1 paper** (Wooden 10,
Stone 9, Iron 16, Gold 6, Diamond 7, Redstone 6, Armor 13 = 67) -- Optimus-3 does not re-list
individual tasks, so the same 3-task/count mismatches vs. our 70 tasks apply here too (see
Optimus-1 section above). Optimus-3 itself is an end-to-end MoE model (Qwen2.5-VL backbone, 6.8B
params, Task/Layer routers) with **no memory-retrieval module** and **no separate controller**
like STEVE-1 -- it departs from the memory-augmented approach of Optimus-1/JARVIS-1. Baselines that
use an MLLM planner (GPT-3.5/4o/Qwen2.5-VL(-SFT)/Gemini-1.5-pro) pair it with STEVE-1 as the
policy.

**Table III** ("Main Result... on Long-Horizon Benchmark"): full per-group SR +/- SD (as
fractions, converted to % in the JSON) for 17 methods including DEPS, JARVIS-1, Optimus-1,
Optimus-2, an Optimus-3-Action ablation, and Optimus-3 (full). **Group-level only -- no per-task
breakdown and no appendix in this paper** (all content fits in Tables I-V, Figs 1-12).

**Table IV** ("Open-ended crafting tasks"): genuine **per-task** CR (completion rate) and SR
numbers, but for a **separate, small (5-task) open-ended/live-inventory benchmark** using
natural-language instructions -- not the 67-task tech-tree benchmark. Mapped to our uids:
wooden_00 (wooden pickaxe, SR 75%), stone_04 (stone sword, SR 70%), iron_12 (iron ingot, SR 65%),
diamond_05 (diamond sword, SR 35%); "golden shovel" (SR 55%) has **no corresponding uid** in our
list (we only have golden pickaxe/axe/sword/hoe). **Do not conflate Table III's tech-tree group SR
with Table IV's open-ended per-task SR** -- they are different experiments/protocols.

---

## Overall caveats

1. **No paper in this set publishes a full per-task SR table covering all 70 of our uids for its
   own full end-to-end pipeline.** The best per-task coverage is Optimus-1 (Appendix F.1, Wood +
   Stone only, ~13 uids) and JARVIS-1 (Tables 5-7, Wood + Stone + Iron largely complete, ~26 uids),
   both cut off by fetch/content-size limits before reaching Gold/Diamond/Redstone/Armor appendix
   tables. MineEvolve and Optimus-3's Table III publish **no per-task numbers at all** (group-level
   only). DEPS publishes per-task numbers only for a separate low-level controller-only benchmark
   (Table 18), not its full LLM-planner pipeline.
2. **Task-count mismatches vs. our 70-task benchmark:** Optimus-1/Optimus-3 use 67 tasks (Wood 9-10
   vs our 11, Stone 8-9 vs our 10, Gold 6 vs our 7); JARVIS-1 uses 200+ tasks across 11 groups
   (broader scope, different grouping, e.g. separate "Blocks"/"Decoration"/"Food" groups we don't
   have); DEPS uses its own MT1-MT8 grouping that mixes tiers and predates gold/redstone tasks
   entirely; MineEvolve's 70-task MCU subset matches our group counts exactly (11/10/16/7/6/7/13)
   but does not publish individual task names, so even that agreement can't be verified 1:1.
3. **No table across all 5 papers was an unreadable image** -- every table cited above is real
   HTML/LaTeXML text that was successfully parsed. All gaps are due to (a) the paper simply not
   publishing per-task data, or (b) a fetch/content-size limitation encountered when trying to
   reach later appendix tables (documented per-paper above), never due to OCR/image failure.
4. Environment/controller assumptions in the original task brief were **not all correct** and have
   been corrected inline per paper: DEPS uses MC-Controller (not STEVE-1) on MineDojo;
   JARVIS-1 uses vanilla Minecraft's native interface (not MineDojo); Optimus-3 uses no
   memory module and no separate low-level controller (it's end-to-end MoE).
5. `match: "approx"` in the JSON always means "same target item/group but the exact paper task
   framing, tier, or quantity differs (or existence in that specific group is inferred rather than
   confirmed)"; `match: "none"` means the item/task has no plausible counterpart in that paper's
   benchmark. `"sr": null` means no numeric value could be retrieved for that mapping, even though
   the mapping itself (or its absence) is noted.
