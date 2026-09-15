# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off helper: clone libero_composite BDDLs and pruned_init pickles
into a parallel libero_composite_vague suite, where the *filename* (and
hence the policy prompt via ``grab_language_from_filename``) is rephrased
to use vague object references / synonyms.

Goal predicates and init states are unchanged, so PSR comparisons between
the two suites isolate the prompt-distribution-shift effect.

Run from the repo root: ``python scripts/_make_libero_composite_vague.py``
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BDDL = REPO / "data/libero_composite/bddl_files"
SRC_INIT = REPO / "data/libero_composite/init_files"
DST_BDDL = REPO / "data/libero_composite_vague/bddl_files"
DST_INIT = REPO / "data/libero_composite_vague/init_files"

# (original_basename_no_ext, new_basename_no_ext, new_language_string)
PAIRS: list[tuple[str, str, str]] = [
    (
        "KITCHEN_SCENE3_first_turn_on_the_stove_then_put_the_frypan_on_the_stove_then_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_cooking_surface_and_place_both_pieces_of_cookware_on_it",
        "turn on the cooking surface and place both pieces of cookware on it",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_black_bowl_then_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_put_both_items_into_the_bottom_drawer_and_shut_it",
        "put both items into the bottom drawer and shut it",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_butter_then_the_chocolate_pudding_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_stash_both_food_items_into_the_top_drawer_and_close_it",
        "stash both food items into the top drawer and close it",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_alphabet_soup_then_the_cream_cheese_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_put_three_grocery_items_into_the_basket",
        "put three grocery items into the basket",
    ),
    (
        "STUDY_SCENE1_first_put_the_book_in_the_front_compartment_then_put_the_mug_to_the_right_of_the_caddy",
        "STUDY_SCENE1_put_the_reading_material_in_the_front_compartment_and_the_cup_to_the_right",
        "put the reading material in the front compartment and the cup to the right",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_put_the_drink_into_the_bottom_drawer_and_shut_it",
        "put the drink into the bottom drawer and shut it",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_black_bowl_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_put_the_dish_in_the_top_drawer_and_close_it",
        "put the dish in the top drawer and close it",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_ketchup_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_put_both_red_condiments_in_the_basket",
        "put both red condiments in the basket",
    ),
    (
        "LIVING_ROOM_SCENE5_first_put_the_red_mug_on_the_left_plate_then_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_and_the_other_mug_on_the_right",
        "put the red mug on the left and the other mug on the right",
    ),
    (
        "LIVING_ROOM_SCENE6_first_put_the_red_mug_on_the_plate_then_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_cup_on_the_dish_and_the_dessert_to_its_left",
        "put the cup on the dish and the dessert to its left",
    ),
]


def main() -> None:
    DST_BDDL.mkdir(parents=True, exist_ok=True)
    DST_INIT.mkdir(parents=True, exist_ok=True)
    for old, new, language in PAIRS:
        src = SRC_BDDL / f"{old}.bddl"
        dst = DST_BDDL / f"{new}.bddl"
        text = src.read_text()
        # Replace the (:language ...) line with the vague phrasing so the
        # BDDL itself is internally consistent (some downstream tools
        # read it).  The eval prompt is determined by the filename
        # via grab_language_from_filename, which already matches `new`.
        text = re.sub(
            r"\(:language [^\n]+\)",
            f"(:language {language})",
            text,
            count=1,
        )
        dst.write_text(text)
        print(f"  bddl: {dst.name}")
        # init pickle: same content, new name
        src_init = SRC_INIT / f"{old}.pruned_init"
        dst_init = DST_INIT / f"{new}.pruned_init"
        shutil.copy2(src_init, dst_init)
        print(f"  init: {dst_init.name}")
    print(f"\nDone: {len(PAIRS)} vague composites in {DST_BDDL.parent}")


if __name__ == "__main__":
    main()
