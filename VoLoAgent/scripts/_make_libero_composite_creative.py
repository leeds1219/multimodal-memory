# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clone libero_composite BDDLs/inits into a parallel
libero_composite_creative suite, where the *filename* (and hence the
policy prompt via ``grab_language_from_filename``) expresses the
goal/result-state rather than spelling out actions.  Tests whether
intent-style prompts (e.g. "warm up both pieces of cookware") need
scene-reasoning that the orchestrator can supply.

Run from repo root: ``python scripts/_make_libero_composite_creative.py``
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BDDL = REPO / "data/libero_composite/bddl_files"
SRC_INIT = REPO / "data/libero_composite/init_files"
DST_BDDL = REPO / "data/libero_composite_creative/bddl_files"
DST_INIT = REPO / "data/libero_composite_creative/init_files"

PAIRS: list[tuple[str, str, str]] = [
    (
        "KITCHEN_SCENE3_first_turn_on_the_stove_then_put_the_frypan_on_the_stove_then_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_warm_up_both_pieces_of_cookware_on_the_stove",
        "warm up both pieces of cookware on the stove",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_black_bowl_then_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_tidy_the_table_by_storing_both_items_in_the_bottom_drawer_and_closing_it",
        "tidy the table by storing both items in the bottom drawer and closing it",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_butter_then_the_chocolate_pudding_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_refrigerate_the_dairy_and_the_dessert_in_the_top_drawer_and_close_it",
        "refrigerate the dairy and the dessert in the top drawer and close it",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_alphabet_soup_then_the_cream_cheese_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_pack_three_groceries_in_the_basket_leaving_the_ketchup",
        "pack three groceries in the basket leaving the ketchup",
    ),
    (
        "STUDY_SCENE1_first_put_the_book_in_the_front_compartment_then_put_the_mug_to_the_right_of_the_caddy",
        "STUDY_SCENE1_set_up_the_desk_with_the_book_in_front_and_the_cup_to_the_right",
        "set up the desk with the book in front and the cup to the right",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_store_the_wine_in_the_bottom_drawer",
        "store the wine in the bottom drawer",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_black_bowl_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_put_the_dish_away_in_the_top_drawer",
        "put the dish away in the top drawer",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_ketchup_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_pack_both_red_bottles_in_the_basket",
        "pack both red bottles in the basket",
    ),
    (
        "LIVING_ROOM_SCENE5_first_put_the_red_mug_on_the_left_plate_then_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_give_each_plate_a_mug_with_the_red_one_on_the_left",
        "give each plate a mug with the red one on the left",
    ),
    (
        "LIVING_ROOM_SCENE6_first_put_the_red_mug_on_the_plate_then_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_place_the_dessert_to_the_left_of_the_cup_on_the_plate",
        "place the dessert to the left of the cup on the plate",
    ),
]


def main() -> None:
    DST_BDDL.mkdir(parents=True, exist_ok=True)
    DST_INIT.mkdir(parents=True, exist_ok=True)
    for old, new, language in PAIRS:
        src = SRC_BDDL / f"{old}.bddl"
        dst = DST_BDDL / f"{new}.bddl"
        text = src.read_text()
        text = re.sub(
            r"\(:language [^\n]+\)",
            f"(:language {language})",
            text,
            count=1,
        )
        dst.write_text(text)
        print(f"  bddl: {dst.name}")
        shutil.copy2(SRC_INIT / f"{old}.pruned_init", DST_INIT / f"{new}.pruned_init")
        print(f"  init: {new}.pruned_init")
    print(f"\nDone: {len(PAIRS)} creative composites in {DST_BDDL.parent}")


if __name__ == "__main__":
    main()
