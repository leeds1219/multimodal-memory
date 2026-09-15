# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror libero_composite into a libero_composite_creative_v2 suite
with stronger scenario / intent prompts (e.g. 'start preparing
breakfast', 'put the leftovers in the fridge', 'pack a picnic basket').
The goal predicates and inits are unchanged; only the *filename*
(which becomes the policy prompt via grab_language_from_filename)
varies.  Designed to push prompts past pi05's training distribution
so VLM disambiguation has more room to win.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BDDL = REPO / "data/libero_composite/bddl_files"
SRC_INIT = REPO / "data/libero_composite/init_files"
DST_BDDL = REPO / "data/libero_composite_creative_v2/bddl_files"
DST_INIT = REPO / "data/libero_composite_creative_v2/init_files"

PAIRS: list[tuple[str, str, str]] = [
    (
        "KITCHEN_SCENE3_first_turn_on_the_stove_then_put_the_frypan_on_the_stove_then_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_start_preparing_breakfast_on_the_stove",
        "start preparing breakfast on the stove",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_black_bowl_then_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_clean_up_the_table_into_the_bottom_drawer_and_shut_it",
        "clean up the table into the bottom drawer and shut it",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_butter_then_the_chocolate_pudding_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_put_the_leftovers_in_the_fridge_and_shut_it",
        "put the leftovers in the fridge and shut it",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_alphabet_soup_then_the_cream_cheese_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_pack_a_picnic_basket_but_leave_the_ketchup_behind",
        "pack a picnic basket but leave the ketchup behind",
    ),
    (
        "STUDY_SCENE1_first_put_the_book_in_the_front_compartment_then_put_the_mug_to_the_right_of_the_caddy",
        "STUDY_SCENE1_set_up_the_study_spot_book_in_front_and_drink_on_the_right",
        "set up the study spot, book in front and drink on the right",
    ),
    (
        "KITCHEN_SCENE4_first_put_the_wine_bottle_in_the_bottom_drawer_then_close_it",
        "KITCHEN_SCENE4_stow_the_wine_for_the_night",
        "stow the wine for the night",
    ),
    (
        "KITCHEN_SCENE10_first_put_the_black_bowl_in_the_top_drawer_then_close_it",
        "KITCHEN_SCENE10_put_the_bowl_out_of_sight_in_the_cabinet",
        "put the bowl out of sight in the cabinet",
    ),
    (
        "LIVING_ROOM_SCENE1_first_put_the_ketchup_then_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE1_load_the_basket_with_both_condiments",
        "load the basket with both condiments",
    ),
    (
        "LIVING_ROOM_SCENE5_first_put_the_red_mug_on_the_left_plate_then_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_set_the_table_for_tea_with_the_red_mug_on_the_left",
        "set the table for tea with the red mug on the left",
    ),
    (
        "LIVING_ROOM_SCENE6_first_put_the_red_mug_on_the_plate_then_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_serve_afternoon_tea_with_dessert_on_the_left",
        "serve afternoon tea with dessert on the left",
    ),
]


def main() -> None:
    DST_BDDL.mkdir(parents=True, exist_ok=True)
    DST_INIT.mkdir(parents=True, exist_ok=True)
    for old, new, language in PAIRS:
        text = (SRC_BDDL / f"{old}.bddl").read_text()
        text = re.sub(
            r"\(:language [^\n]+\)",
            f"(:language {language})",
            text,
            count=1,
        )
        (DST_BDDL / f"{new}.bddl").write_text(text)
        shutil.copy2(
            SRC_INIT / f"{old}.pruned_init",
            DST_INIT / f"{new}.pruned_init",
        )
        print(f"  {new}")
    print(f"\nDone: {len(PAIRS)} v2-creative composites in {DST_BDDL.parent}")


if __name__ == "__main__":
    main()
