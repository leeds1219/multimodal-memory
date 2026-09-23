# Stage B: port plan for JARVIS-1 and DEPS on MineRL 1.0 + STEVE-1 + Gemini

Status: analysis only. No code has been written or run. Sources:
- JARVIS-1 code: `/home/rag/data/official/JARVIS-1` (the offline-evaluation release)
- JARVIS-1 paper: arXiv 2311.05997 (HTML version: Sec. 3, App. A.1–A.3, B, Table 2)
- DEPS code: `/home/rag/data/official/MC-Planner`
- DEPS paper: arXiv 2302.01560v3 (Sec. 3, Tables 1–4, 18, 19, App. C)
- MineEvolve paper: arXiv 2603.13131v3, plus the code at `/home/rag/data/official/MC-MineEvolve`
- Our stack: `baseline-repro/configs/{llm,tasks,seeds,task_orders}.yaml`, `baseline-repro/llm/gemini_client.py`

Throughout: **[C]** means the released code, **[P]** means the paper, **[ME]** means the MineEvolve paper or repo.

---

## 0. Headline findings

1. **The JARVIS-1 release makes no planning LLM calls.** `offline_evaluation.py` finds the task by exact name in `jarvis/assets/memory.json` and runs that stored plan open-loop. There is no query generation, no planner, no self-check, no self-explain and no replanning. If the task is not in memory it raises `NotImplementedError("Online generating plan ... not merged yet")`. The only LLM call is `core.get_skill()`: a gpt-3.5-turbo few-shot prompt that picks one STEVE-1 text prompt, or an equip action, out of `skill.json` for a mine subgoal. So "run the released code" and "run JARVIS-1 as in the paper" are two different baselines. The planner part has to be rebuilt from the paper's App. A.2 prompts.
2. **`memory.json` holds 188 successful plans for 60 of our 70 tasks**, keyed by the exact item name. Run offline, JARVIS-1 is close to an oracle-plan baseline on those 60 tasks. The paper-faithful setting is the same: retrieval with the paper's 425-entry memory, which covered the evaluated tasks. Report which mode we use.
3. **DEPS code has no Selector.** `selector.py` is a stub where every method raises `NotImplementedError`. The horizon head inside the MineDojo controller predicts a ranking, but `main.py` throws it away (`_ranking, _action = ...`). The prompts produce linear plans with no parallel goals, so goals run in a fixed order. The released code is really "DEP" (the paper's "w/o Selector" ablation). Its trained horizon-predictive selector (Impala CNN, trained on MineDojo trajectories) has no released weights and no STEVE-1/MineRL counterpart.
4. **The DEPS paper already reports a MineRL 1.16.5 + STEVE-1 variant** (Table 3): MT1 84.05, MT2 80.32, MT3 24.25, MT4 36.21, MT5 9.16, MT6 17.22, MT7 16.79, MT8 1.84. That is the closest published precedent for our port.
5. **The DEPS code used a different model for each role.** Planner, describe and explain calls went to `code-davinci-002` (Codex) as text completion at T=0.7 with `stop=["Human:"]`. The goal parser went to `text-davinci-003` at T=0, one call per plan line, with the prompt truncated to the last 4000 characters. Both models are retired. With Gemini, every role becomes a chat call that emulates text completion.
6. **The paper and code disagree on craft and smelt for both methods.** Both papers say craft and smelt are *functional* MineDojo-style actions. The JARVIS-1 code actually drives a scripted **GUI** mouse agent (`assembly/scripts/craft_agent.py`, `smelt_agent.py`), which needs per-slot inventory info from a patched MCP-Reborn (`scripts/mcp_patch.diff`, about 16k lines). The DEPS code uses MineDojo's functional `craft` action (`act[5]=4, act[6]=item_idx`) plus scripted placement of the crafting table and furnace.
7. **MineEvolve's own `executor/craft_helper.py` does not craft.** It plays a sound through a chat command and polls the inventory. So the shared stack needs a real craft/smelt helper (Section 3). This decides how every method scores on crafting.
8. **Several MineEvolve baseline numbers match the original papers almost exactly** (Section 4). Treat them as reference points only, not as numbers we should expect to reproduce.

---

## 1. JARVIS-1

### 1.1 Components: paper vs. release

| Component (paper) | In release? | Notes |
|---|---|---|
| MLM = GPT (ChatGPT/GPT-4) plus MineCLIP "descriptor": about 1000 wiki-derived situation sentences, those above a MineCLIP similarity threshold added to the prompt | **No** | README: "multimodal descriptor ... not released". The sentence bank is not released either. |
| Symbolic state to text (inventory, biome, position) using templates | Partial | `core.translate_inventory/equipment/height` (used only by the skill selector). |
| Query generation (backward-search LLM reasoning, bounded depth, produces sub-task text queries) | **No** | Paper Fig. 5 and Sec. 3.3 only. No prompt is given in the paper. |
| Multimodal retrieval: CLIP-text match of the task key above a threshold, then rank by `CLIP_v(s_z)ᵀCLIP_v(s_x)`, **top-1 per sub-goal** (Fig. 5) | **No** | Memory has no states or images (README: "remove the multimodal state"). Only an `image` filename string remains, and no image files ship. |
| Planning prompt (App. A.2 Prompt 1) | **No** (text in paper only) | `assets/prompts.json` is an empty file (0 bytes). |
| Goal-parsing prompt (Prompt 2) | No | Paper only. |
| Self-check (Prompt 4) | **Removed** | README: "Remove the self-check module". |
| Self-explain (Prompt 3) and replanning | No | Paper only. |
| Skill selection for mine goals (LLM picks a STEVE-1 prompt or an equip) | **Yes** | `assembly/core.py:get_skill` with `assets/skill.json` (45 items). Not described in the paper. |
| Controller: STEVE-1 text conditioning for mine | Yes | `steveI/steveI_text.py`, cond_scale 6.0; hotbar keys masked; GUI force-closed. |
| Craft/smelt/equip | Yes (scripted GUI) | `assembly/scripts/*`. The paper says functional actions (App. A.1, B.2). |
| Memory | Yes (fixed, text-only) | `assets/memory.json`: 188 entries `{task: {plan:[{goal:{item:n}, type: mine/craft/smelt, text}], init_inventory, status:'success', image, time}}`. |
| Self-instruct and lifelong learning (memory growth, 4 epochs, 425 entries) | **No** | README: `learning.py` / online evaluation "coming soon". |
| Task suite | Yes | `assets/tasks.json`: 185 tasks with biome, mobs and init_inventory (all empty). The env YAML always gives an **iron_axe in slot 35**. |

### 1.2 Prompts and files to reuse verbatim

- **Skill-selection prompt (code):** `jarvis/assembly/core.py` lines 50–95. The system message plus three few-shot pairs are copied verbatim. The user query format is `Task: Obtain {item}.\nSkills: 1. ..., 2. ...,\nAgent State: {inv} {equip} {height}`. Original parameters: `temperature=1, max_tokens=256, top_p=1`. The parser takes `Action: <int>` and falls back to a random skill.
- **`jarvis/assets/skill.json`**: item → candidate STEVE-1 text prompts or equip actions. Items not listed fall back to `"get {item}"` (for example `sugar_cane`, `jungle_log`, and `gold_ore`).
- **`jarvis/assets/memory.json`**: the retrieval library (and the offline plans).
- **`jarvis/assets/tag_items.json`**: tag resolution for `logs`, `planks` and similar. `evaluate.monitor_function` counts the **max over tag members, not their sum**; reuse that exact logic.
- **`jarvis/assets/recipes/*.json`**: 1.16 recipes, used by the craft helper.
- **Paper prompts (App. A.2), copy verbatim from arXiv HTML:**
  - Prompt 1 (planning). System: "You are a helper agent in Minecraft. You need to generate the sequences of goals for a certain task in Minecraft. Just refer the history dialogue to give the plan consist of template. Do not explain or give any other instruction." Then `==========` separators and a few-shot dialogue (wooden_axe example, code-style `mine(obj=..., tool=None)` / `craft(obj=..., materials=..., tool=...)` with `# step k:` comments, followed by `User: [Description] I succeed in step 1, 2, ...`). The query is `User: My current inventory has <inventory>. <visual observation>. How to obtain 1 <item> in Minecraft step-by-step?`
  - Prompt 2 (goal parsing): `name / text condition / action / object_item / object_number / materials / tool / rank`.
  - Prompt 3 (self-explain): "Here are some actions that the agent fails to perform in Minecraft. Please give the explanation of action execution failure according to the current inventory information of the agent." Three `Failed Action / Current Inventory` shots.
  - Prompt 4 (self-check): "You are a helper agent in Minecraft. Check the plan whether can be finished." A step-by-step inventory simulation ending in `Return: Step k will failed because of ...`.
  - The paper gives **no prompt** for query generation or for the replan turn after self-explain. We must write these and label them as ours.
- Save the extracted prompt text to `baseline-repro/baselines/jarvis1/prompts/` once. The HTML is in the session scratchpad as `j1.txt`, lines 700–790; re-fetch the arXiv HTML because the scratchpad is temporary.

### 1.3 Loop structure

**Code (`offline_evaluation.evaluate_task`):**
```
plan = memory[task]['plan']                          # 0 LLM calls
for subgoal in plan (index advances only when the subgoal's inventory target is met):
    while not monitor(subgoal.goal):
        mine  -> skill = get_skill(item, info)  (1 LLM call if the item has >1 skill, else 0)
                 if skill.type == 'mine': STEVE-1(text, timeout=600 steps, early stop when monitor true)
                 else: scripted equip(object_item)
        craft/smelt -> scripted GUI craft/smelt(target, target_num)   # failure is only a return flag
        break if the task target is met or total steps > 1200*time_min
return success | timeout
```
- Nothing moves the plan forward except meeting the subgoal's inventory count. A failed craft is retried forever until timeout; there is no replanning. The per-mine-attempt horizon is 600 steps. The episode horizon is 1200 × `--time` (default 10 min = 12000 steps).
- The paper's per-group limits (Table 2) are Wood/Stone/Iron 12k steps and Diamond 36k. For ObtainDiamondPickaxe it uses 20 min and 60 min.

**Paper (Sec. 3, Fig. 3–5), reconstructed:**
```
obs_text = describe(obs)                         # MineCLIP sentence retrieval + templates (no LLM in paper)
subtasks = query_gen(task, obs_text)             # 1 LLM call (backward reasoning, bounded depth)
refs = retrieve(subtasks, obs)                   # top-1 entry per sub-goal; text threshold then visual rank
plan = plan_llm(Prompt1 + refs + state)          # 1 call
loop self_check(plan) -> fix plan                # Prompt 4; 1 call per check, plus 1 replan if a bug is found
for goal in parse(plan):                         # Prompt 2 (LLM parse; 1 call per plan or per line)
    execute(goal)
    on failure: self_explain(Prompt3) -> replan  # 2 calls per round; paper: "usually only 2-3 rounds"
on success: memory.add({task, state, plan})      # learning stage only
```

### 1.4 Memory

| | Paper | Code |
|---|---|---|
| Stored | `{task, state (visual obs + symbolic), plan}`; multiple entries per task allowed | `{plan, init_inventory, status, time, image(name only)}`, one per task, 188 tasks |
| Key | multimodal (task text + observation) | task name string |
| Retrieval | LLM query-gen → CLIP-text on task keys (threshold) → CLIP-visual rank → top-1 per sub-goal | exact dict lookup `memory[task]` |
| Growth | built in a learning stage (self-instruct, 4 epochs → 425 entries), then **frozen** for evaluation ("we use this as the default setting"); the Fig. 7 ablation varies its size | none (online and `learning.py` not released) |

**Port decision (recommended):** use a frozen memory loaded from `memory.json`, which is faithful to the paper's evaluation protocol. Retrieval is LLM query-gen followed by **text-only** matching of sub-task names to memory keys, keeping top-1 per sub-goal (capped by the shared top-K). Visual ranking cannot be done because no states are stored. If several entries share a task, keep the first; the release has one per task anyway. Do **not** grow memory during evaluation, which matches [ME] Table 5 ("continual knowledge update: No"). Optional ablation: text memory without query-gen ("Text Memory" in Fig. 8).

### 1.5 Paper vs. code disagreements
1. Planner: the paper plans online with an LLM; the code replays stored plans (0 planning calls).
2. Self-check: present in the paper, removed in the code. Self-explain and replan: in the paper, absent from the code.
3. Craft/smelt: functional actions in the paper; scripted GUI mouse agent (patched MCP-Reborn) in the code.
4. Memory: multimodal keys with visual ranking in the paper; exact text lookup with no states in the code.
5. Plan format: code-style `mine(obj={...}, tool=...)` in the paper; structured JSON `{goal, type, text}` in memory. Retrieved plans must be **rendered into the Prompt-1 code style** to serve as few-shot references (our renderer, a deviation).
6. The skill-selector LLM (gpt-3.5) and `skill.json` are in the code but not the paper.
7. Initial state: the code's env YAML gives an `iron_axe` to every task. The paper says an iron_axe is given only for some groups.
8. The paper says "ChatGPT and GPT-4" (defaults, temperature unchanged). The code uses `gpt-3.5-turbo`, T=1, for skill selection only.

### 1.6 Glue for MineRL 1.0 + STEVE-1
- **Controller:** our STEVE-1 (same VPT + prior + MineCLIP stack as `steveI_text.py`), cond_scale 6.0. Keep the hotbar mask and GUI-close override from `SteveIText.do`. Mine timeout 600 steps per attempt, early stop when `monitor_function(goal)` is true.
- **Inventory/info adapter:** MineRL 1.0 `obs['inventory']` is an **aggregated** `{item: count}` dict, while JARVIS expects `info['inventory'][0..35]` slots, `equipped_items.mainhand.type`, `player_pos.y` and `isGuiOpen`. Write `info_adapter()` that builds pseudo-slots from counts (enough for `monitor_function`, `translate_*`), takes y and mainhand from `location_stats` / `equipped_items`, and tracks GUI state ourselves. Per-slot data is needed only by the GUI craft script, which we replace.
- **Craft/smelt/equip helper (unavoidable deviation):** we cannot use the JARVIS GUI script as-is, because it needs MCP-Reborn per-slot info and fixed pixel slot coordinates. Two options, one of which must be chosen for **all** methods:
  (a) A GUI mouse-macro craft agent adapted to MineRL 1.0's 640×360 GUI. It would be ported from `craft_agent.py` coordinates, which were also 640×360. This is the most faithful to the JARVIS code and costs env steps. Slot identity must be inferred by tracking our own clicks.
  (b) A functional "recipe check + consume + add" via chat commands (`/clear`, `/give` of the result only when the recipe inputs are present and the table/furnace precondition holds). This is faithful to both **papers** (functional actions) and to DEPS code semantics. It costs a fixed step budget (for example the DEPS-like 150/200-step caps as timeouts). Recommendation: (b), with a documented step charge, used by all methods.
- Equip: `hotbar.N` swap, or a `/replaceitem`-free approach: move the item to a hotbar slot through the inventory GUI, or allow a chat-command swap of held items. Document the choice.
- **Tag/name mapping (1.16 names):** `logs`/`planks` tags are handled through `tag_items.json`. Our tasks use `oak_log`/`oak_planks`. MineRL 1.0 is 1.16, so the names match.
- **LLM:** route through `GeminiClient` (model forced to gemini-3-flash-preview). Keep the original temperature and max_tokens where the code sets them (skill selection: T=1, max_tokens=256). For paper prompts, use the provider default (paper: "all hyper-parameters ... default").
- **Visual observation text:** paper-faithful would be MineCLIP similarity against a 1000-sentence bank we do not have. Choose between:
  (i) passing the current POV image to Gemini inside the planning call, which is simpler and uses the multimodal LLM we already have (deviation: an end-to-end MLM, which the paper says it avoided);
  (ii) building our own small sentence bank of biome and visible-entity phrases, scored with our MineCLIP. This is closer to the paper.
  Recommend (ii) as primary if time allows, else (i) with the prompt text `<visual observation>` replaced by a one-line Gemini caption. Mark it as a deviation either way.
- **Env init:** empty inventory per [ME] and our `tasks.yaml`. Drop JARVIS's default iron_axe and document that (the DEPS paper gives an axe; see below). Keep our shared reset commands (night vision, keepInventory, no daylight cycle).
- **Horizon:** use our shared per-group minutes (wooden 2, stone 3, iron 20, gold 10, redstone 15, diamond 30, armor 25) × 1200 steps/min, not JARVIS's 12k/36k.

### 1.7 LLM calls per episode (paper-faithful port)
- Up front: query-gen 1 + plan 1 + self-check 1 (+1 fix if flagged) + parse 1 (whole plan in one call; per-line would be N) = **3–5**.
- Per replan round: explain 1 + replan 1 + self-check 1 + parse 1 = 4. The paper says 2–3 rounds are typical.
- Skill selection: 1 per mine attempt whose item has more than one skill. Easy tasks 1–3; iron+ tasks 5–20, because each 600-step STEVE-1 attempt on iron_ore, diamond or cobblestone calls again.
- Estimate: **easy ~5–10, hard ~15–35 calls per episode.** Offline-code mode: 0 planning calls plus skill selection only (about 1–20).

### 1.8 Risks for JARVIS-1
- Oracle-plan leakage: memory contains the test tasks. Paper-faithful, but it inflates results against DEPS. Report a "memory excludes the exact task" ablation if budget allows.
- Tasks with no memory entry (10 of 70): `sapling`, `cobblestone` (×2 tasks), `iron_axe`, `golden_carrot`, `repeater`, `comparator`, `turtle_helmet`, `leather` (kill cow), plus `logs`-style wood tasks, which are covered by `oak_log`. These need a real LLM plan and cannot run offline.
- Some MCU tasks cannot be done by either method's action set: villager trading (iron_15), washing a chestplate in a cauldron (armor_10), repairing at a smithing table (armor_11), turtle helmet (armor_09, needs scutes). Expect 0% and keep them in the denominator.
- Prompt gaps: query-gen and replan-turn prompts are unpublished and must be authored by us.
- The GUI crafting route is fragile under MineRL 1.0. The functional route is a deviation from the code.

---

## 2. DEPS (MC-Planner)

### 2.1 Components: paper vs. release

| Component (paper) | In release? | Notes |
|---|---|---|
| **Describe** (symbolic info → text: inventory, biome, success/failure of step k) | Yes | `planner.generate_success_description / generate_failure_description / generate_inventory_description`. Inventory only; biome is not used in the code. `diamond_axe` is filtered out (the given tool). |
| **Explain** (LLM chain-of-thought on why the step failed) | Yes | `generate_explanation()`: Codex continues the dialogue with `AI: Because ...`. |
| **Plan** (code-style plan, interactive dialogue) | Yes | `initial_planning()` and `replan()`. |
| **Select** (horizon-predictive selector over parallel goals, trained Impala CNN, trained with the controller) | **No** | `selector.py` is a stub. The controller's `pred_horizon` is computed and discarded. The prompts never produce parallel goals. |
| Goal parser (LLM) | Yes | `data/parse_prompt.txt` with `text-davinci-003`, **one call per plan line** containing `#`. |
| Goal library (goal → type, preconditions, tool) | Yes | `data/goal_lib.json` (86 goals: 73 craft, 7 mine, 6 smelt). |
| Success detector | Yes | Inventory count check (`check_inventory`, cumulative count, requires `goal_eps>1`). |
| Controller: learned mine policy for log/sheep/cow/pig; **scripted dig-down** for stone/cobblestone/coal/iron_ore/diamond; functional craft/smelt with scripted table/furnace placement | Yes | `controller.py`: `MineAgentWrapper.script_goals`, `CraftAgent`. Needs MineDojo fork (MC-Simulator) and MC-Controller checkpoint. |
| Task suite (71 tasks, MT1–MT8, 3000/6000/12000 steps, empty inventory plus an axe) | Yes (70 in `task_info.json`) | Group prompts `data/group_prompts/` are referenced but commented out and absent. |

### 2.2 Prompts and files to reuse verbatim
- `data/task_prompt.txt`: the 2-line preamble. Keep the typo "certPlannern" verbatim.
- `data/deps_prompt.txt`: the full iron_pickaxe interactive few-shot dialogue (plan, fail, inventory, explanation, replan, ..., success). It is used both as the initial prompt and as the replan context: `task_prompt + deps_prompt + "Human: {question}\n"`.
- `data/parse_prompt.txt`: 4-shot parser. The input is `input: {line}`. The output fields are `name/action/object/tool/rank`, parsed by line prefix, and `object` is parsed with `eval()`; use `ast.literal_eval` instead.
- `data/goal_lib.json`: preconditions and tools for the precondition-violation replan trigger and table/furnace use.
- `data/goal_mapping.json` (`horizon`/`mineclip` maps): mine item → controller goal text. Use it as the basis for STEVE-1 prompts (`log`→"mine wood" / "chop down the tree", `wool`→"kill sheep", and so on).
- `data/task_info.json`: question template `"How to obtain {item}?"`.
- Dialogue strings from `planner.py`, verbatim: `Human: I succeed on step {k}.\n`, `Human: I fail on step {k}` (no newline, and the LLM completes it), `Human: My inventory now has {n} {item}, ...\n`, `Human: Please fix above errors and replan the task '{question}'.\n`.

### 2.3 Loop structure (code, `main.Evaluator`)
```
reset: plan = LLM(task_prompt + deps_prompt + "Human: How to obtain X?\n")   # 1 call (Codex, T=0.7, max_tokens 1024, stop "Human:")
       goals = [parse(line) for line in plan if '#' in line]                # N calls (davinci-003, T=0, 256 tok)
       unknown goals: map by output item via goal_lib, else drop; empty plan -> default mine_log
for t in range(max_ep_len):                                                 # 3000/6000/12000
    if inventory >= curr_goal.object and goal_eps > 1: describe success, pop goal
    act: craft/smelt -> scripted craft agent; mine -> controller(goal_mapping[item])
    auto-equip a required pickaxe/axe from goal precondition
    replan if: (mine and precondition/tool missing from inventory)          # checked every step
            or (craft and goal_eps > 150) or (smelt and goal_eps > 200)
      replan = fail-describe (1 call) + inventory text + explain (1 call) + replan (1 call) + parse (N calls)
    stop when replan_rounds > 12 (failure) or the task item is in inventory (success)
```
- Mine goals have **no timeout** in the code. They run until the task horizon unless a precondition is missing.
- The dialogue grows with every round and is sent in full each time (Codex window). `query_gpt3` truncates to the last 4000 characters (parser only).
- API retry: up to 10 attempts per call with key rotation.
- The paper says Codex allowed about 7–8 rounds ("∞" in Table 4). The code caps at 12.

### 2.4 Memory
No cross-episode memory in the paper or the code. The only state is the within-episode dialogue (plan history, descriptions, explanations). It resets per episode. Nothing grows at test time.

### 2.5 Paper vs. code disagreements
1. **Selector:** a core component in the paper (+22–33% in the parallel-goal ablation; full DEPS vs. DEP MT1 79.8 vs. 75.7). Not wired in the code, and the prompts have no parallel goals.
2. Descriptor: the paper says inventory plus biome; the code uses inventory only.
3. LLM: the paper uses Codex for Minecraft. The code also uses text-davinci-003 for parsing, which the paper's "Goal Parser" mentions without naming a model. Table 19 shows the choice of LLM matters little for final DEPS success (MT1: Codex 79.8, GPT-3 75.4, ChatGPT 70.2, GPT-4 89.3).
4. Replan cap: about 7–8 rounds (token-limited) in the paper, 12 rounds in the code.
5. Controller: the paper describes an IL goal-conditioned policy with 262 goals and Table 18 success rates. The code scripts dig-down goals and uses the network only for 4 goals.
6. Paper Table 1: every task starts with an empty inventory **plus an axe**. The code environment is not in this repo (MC-Simulator), but the `diamond_axe` filter in the inventory description suggests the axe was a diamond_axe.

### 2.6 Glue for MineRL 1.0 + STEVE-1
- **Chat-completion emulation of text completion:** send `system: "Continue the following transcript exactly in its format. Output only the continuation."` (ours) and `user: <full transcript>`. Honor the stop sequence `"Human:"` by cutting client-side, and pass `stop` if Gemini's OpenAI endpoint accepts it. Keep T=0.7 and max_tokens=1024 for planning/explaining, and T=0 with max_tokens=256 for parsing, **but** thinking tokens on gemini-3-flash may use up max_tokens. Either raise max_tokens and trim the output, or set the reasoning effort low. Document this.
- **Parser:** keep the LLM parser for fidelity (N calls per plan). As a budget variant, a deterministic regex parser of `mine({...}, tool)` / `craft({...}, {...}, table)` / `smelt(...)` gives the same fields. Use it only if the shared LLM-call budget requires it, and label it.
- **Mine goals → STEVE-1 text:** `log`→"chop down the tree"; `cobblestone`/`stone`→"dig down and break stone blocks"; `iron_ore`→"dig down" then "mine iron ore" (the paper's scripted dig-down becomes STEVE-1 prompts; reuse JARVIS `skill.json` texts for consistency); `wool`/`mutton`→"kill sheep"; `leather`/`beef`→"kill cow"; `porkchop`→"kill pig"; `diamond`→"dig down" / "mine diamond ore". Items outside `goal_mapping` fall back to `"get {item}"` (JARVIS convention). The DEPS goal_lib lacks 20 of our 70 targets: all golden items, redstone chain, `diamond_pickaxe/axe/hoe/sword`, `smithing_table`, `sapling`, `turtle_helmet`. The code **drops** unparseable or unknown goals ("parsed goal is not supported"). For a fair port, extend `goal_lib.json` from 1.16 recipes (JARVIS `assets/recipes`) with the same schema. This is a documented deviation; without it DEPS scores 0 on about 20 tasks by construction.
- **Scripted dig-down (code):** MineRL has no compass/GPS as used in MineDojo, but `location_stats` gives pitch and y. Either port `MineAgentWrapper` dig-down (look down, attack, depth 30 for iron and 10 for diamond; with 1.16 ore distribution diamonds are at y ≤ 16) or use STEVE-1. MineEvolve says DEPS "uses STEVE-1 as the low-level execution policy", and the DEPS paper's MineRL variant uses STEVE-1. **Recommendation: STEVE-1 for all mine goals** (matches DEPS Table 3 and [ME]), and note that the scripted dig-down is dropped.
- **Craft/smelt:** the same shared helper as JARVIS (Section 1.6, option b is closest to the MineDojo functional craft). Keep the DEPS failure semantics: a craft not done within 150 steps, or a smelt within 200, triggers a replan. With a functional helper that fails instantly when materials are missing, charge the step budget or declare failure at once. Declaring failure at once is closer to the intent; document it.
- **Auto-equip:** before each step, if the goal precondition has a pickaxe or axe that is not in the main hand, swap it (hotbar key in MineRL).
- **Given tool:** the DEPS paper gives an axe. Our shared protocol says empty inventory. Follow the shared protocol and document the deviation.
- **Inventory description:** from MineRL `obs['inventory']` counts, keep items with count > 0 and order them by first acquisition (the MineDojo slot order cannot be reproduced).
- **Horizon:** the shared per-group minutes instead of 3000/6000/12000.

### 2.7 LLM calls per episode (code-faithful)
- Initial: 1 plan + N parse calls (N ≈ plan lines: 2–5 for wooden tasks, 8–15 for iron/diamond) = **3–16**.
- Per replan round: 3 (fail-describe, explain, replan) + N parse = **6–18**. Cap 12 rounds.
- Estimates: easy (wooden/stone) with 0–3 replans ≈ **5–40**; hard (iron/diamond/armor) often hitting the cap ≈ **60–200**. The dialogue is re-sent every call, so tokens grow roughly quadratically. The deps_prompt alone is about 2.6k tokens.
- A deterministic parser cuts this to 1 + 3×rounds (max 37).

### 2.8 Risks for DEPS
- No selector means the port is really "DEP". Report it that way, or implement a proxy selector. There are no parallel goals in the prompts, so the selector would rarely act even in the code.
- Mine goals with no timeout plus STEVE-1 stalls: one bad mine goal uses the whole horizon. The code does this too; keep it faithful, but log it.
- Gemini behaves differently from Codex on text continuation (possible chattiness or format drift). The parser relies on `'#'` in a line; lines without a comment are silently dropped.
- Extending goal_lib is required for fairness but is a deviation.
- The replan trigger fires every step for mine goals whose precondition (tool) is missing. That can use up 12 rounds within a few steps if the replanned plan keeps the same bad order. The code behaves the same way.

---

## 3. Shared stack decisions (must be identical across methods)
1. **Craft/smelt helper.** MineEvolve's `craft_helper.py` crafts nothing: it plays a sound and polls. Recommended: a functional recipe executor using 1.16 recipes. It checks the materials, needs a crafting table/furnace in the inventory (or placed nearby), consumes the inputs, gives the output via command, and charges a fixed number of env no-op steps. Both papers describe craft and smelt as functional actions, so this is paper-faithful for both.
2. Mine executor: STEVE-1 text with per-attempt timeout (JARVIS 600 steps; DEPS none) → keep each method's own.
3. Equip: hotbar swap.
4. Horizon: `tasks.yaml` minutes × 1200.
5. Success: task item count ≥ 1 in inventory. Non-craftable MCU tasks (trade, wash, repair, turtle helmet) score 0 for every method.
6. LLM budget: [ME] claims a shared "evaluation-time LLM-call budget" but never gives the value. Its own average is 8.31 calls/episode. Code-faithful DEPS goes far past that. Decide whether we enforce a cap. **Recommendation: no cap (faithful). Log calls and cost per episode, and report calls/episode next to SR.**

---

## 4. MineEvolve (arXiv 2603.13131v3): what it says about the baselines
- **Setup:** 70-task MCU tech-tree subset. Wooden 11, Stone 10, Iron 16, Gold 7, Redstone 6, Diamond 7, Armor 13. Empty inventory. Easy groups = Wooden/Stone/Gold; Hard = Iron/Redstone/Diamond/Armor. Task-count-weighted averages. 8×A40.
- **Baselines:** "DEPS, JARVIS-1, Optimus-1 and MineEvolve use STEVE-1 as the low-level execution policy under our interface; Optimus-2 uses its native GOAP policy." "Each method retains its native high-level planning and execution pipeline." Controlled: task set, task–seed split, episode horizon, success criterion, observation/action interface, action primitives, structured state fields, retrieval top-K, memory-token budget, and evaluation-time LLM-call budget. **No numeric values** are given for the horizon, number of runs per task, top-K, token budget or call budget. There are **no per-baseline implementation details** (selector, JARVIS memory, and so on). Table 5 characterizes them: DEPS = "feedback interpretation and subgoal selection"; JARVIS-1 = "weak; mainly successful-case retrieval", no continual update; Optimus-1 = experience pool, partial continual update.
- **Repo** (`MC-MineEvolve/src/mineevolve/conf`): per-group horizons wooden 2 / stone 3 / iron 20 / gold 10 / redstone 15 / diamond 30 / armor 25 min. The config comments say "paper Table 3", but the paper's Table 3 has no horizons. `max_subgoals 12`, `max_steps_per_subgoal 1200`, `top_k 16`, `budget_tokens 512`. The repo contains **no baseline implementations**.
- **Numbers, Table 4 (SR %, Wo/St/Ir/Go/Re/Di/Ar/Overall):**

| Planner | Method | Wo | St | Ir | Go | Re | Di | Ar | Overall |
|---|---|---|---|---|---|---|---|---|---|
| Gemini-3-Flash | DEPS | 84.25 | 69.86 | 17.81 | 4.07 | 9.61 | 2.76 | 4.66 | 29.66 |
| Gemini-3-Flash | JARVIS-1 | 92.99 | 88.58 | 35.21 | 9.79 | 22.96 | 8.92 | 15.52 | 42.04 |
| Gemini-3-Flash | Optimus-1 | 98.70 | 91.76 | 45.92 | 6.85 | 24.48 | 13.06 | 18.98 | 46.73 |
| Gemini-3-Flash | Optimus-2 (GOAP) | 99.10 | 93.55 | 54.20 | 10.05 | 30.25 | 13.80 | 22.35 | 50.45 |
| Gemini-3-Flash | MineEvolve | 98.63 | 93.37 | 55.28 | 13.52 | 33.73 | 12.78 | 27.18 | 52.04 |
| Qwen3.5-Plus | DEPS | 84.05 | 69.34 | 17.39 | 3.47 | 8.94 | 2.43 | 4.36 | 29.25 |
| Qwen3.5-Plus | JARVIS-1 | 93.08 | 88.73 | 35.57 | 11.12 | 26.18 | 9.03 | 15.58 | 42.59 |
| Qwen3.5-Plus | Optimus-1 | 97.91 | 93.60 | 46.06 | 7.05 | 32.03 | 11.42 | 19.09 | 47.42 |
| Qwen3.5-Plus | MineEvolve | 98.71 | 93.48 | 55.83 | 13.79 | 31.24 | 17.06 | 27.63 | 52.52 |
| — | STEVE-1 only | 25.61 | 18.72 | 2.47 | 0.00 | 0.00 | 0.00 | 0.00 | 7.26 |

  Overall, other planners: DEPS 28.90 (Qwen-Flash), 27.93 (GLM-4.7), 31.69 (GPT-5.5). JARVIS-1 41.75 / 40.79 / 44.01. Optimus-1 45.83 / 45.23 / 48.92. MineEvolve 50.09 / 48.43 / 54.93. Runtime (Table 7, 400-episode knowledge base): 8.31 LLM calls per episode, 1216 retrieved tokens.
- **Red flag:** the baseline numbers sit on top of the original papers' figures. DEPS (Qwen-Plus) Wooden 84.05 is **exactly** DEPS paper Table 3 MineRL/STEVE-1 MT1 (84.05). Stone 69.34 vs. JARVIS-1-paper DEPS Stone 69.27. Iron 17.39 vs. 16.92. Diamond 2.43 vs. 2.42. JARVIS-1 Stone 88.73 vs. paper 88.69; Iron 35.57 vs. 34.63; Diamond 9.03 vs. 8.99. The spread across 5 planners is small (about ±2). Our reruns may land well away from these. Treat [ME] baseline numbers as references, not expected values.

---

## 5. Summary of unavoidable deviations (both methods)
1. Gemini (chat) instead of Codex / text-davinci-003 / gpt-3.5 / GPT-4. Text-completion prompts are emulated in chat, and thinking tokens interact with max_tokens.
2. STEVE-1 on MineRL 1.0 (MC 1.16.5) instead of the MineDojo 1.11.2 controller (DEPS). JARVIS-1 also used STEVE-1 on 1.16 (MCP-Reborn), so the gap there is small.
3. A shared functional craft/smelt helper instead of the JARVIS GUI script or MineDojo craft action. Paper-faithful, code-divergent for JARVIS.
4. Empty inventory, no given axe or iron_axe.
5. Shared horizons (MineEvolve config) instead of 3000–12000 (DEPS) or 12k–36k (JARVIS).
6. JARVIS-1: no multimodal descriptor or visual retrieval (never released); self-check and planner prompts come from the paper appendix; query-gen and replan prompts are authored by us; frozen memory from `memory.json`; memory plans are rendered to code style.
7. DEPS: no Selector (as in the code); goal_lib extended to cover the 70 tasks; dig-down done by STEVE-1 instead of a script; biome not described (as in the code).

## 6. Blockers and open decisions
- **B1 (decide):** craft helper design (functional vs. GUI) and its step cost, shared by all methods. MineEvolve's helper is a no-op.
- **B2 (decide):** whether to cap LLM calls. DEPS code-faithful can reach about 200 calls/episode on hard tasks. MineEvolve never states its budget.
- **B3 (decide):** JARVIS-1 mode: paper-faithful online planner with frozen memory retrieval (recommended) vs. released offline replay (oracle plans, fails on 10 tasks).
- **B4 (content):** unpublished JARVIS-1 prompts (query-gen, post-explain replan) and the missing MineCLIP sentence bank.
- **B5 (infra):** STEVE-1 weights and MineRL 1.0 env are assumed to be available from Stage A. Disk is at 100% (12 GB free), so do not download MineDojo, MC-Simulator or the MC-Controller checkpoints; none are needed for this port.
- **B6 (fairness):** 4 MCU tasks (trade, wash, repair, turtle helmet) cannot be completed with craft/mine/smelt primitives by any baseline.
