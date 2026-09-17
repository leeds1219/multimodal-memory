"""Chat-action handler for issuing in-game commands.

Used at episode reset to apply quality-of-life game rules (night vision,
keep-inventory, day-only) and at runtime by the world-shaping wrapper to
spawn ore blocks at the agent's feet when the agent reaches the appropriate
Y-band. NEVER used to grant the agent any item.
"""

from __future__ import annotations

from minerl.herobraine.hero.handlers.agent.action import Action

from minerl.herobraine.hero import spaces


class _ChatText(spaces.Text):
    """MineRL 1.0's ``Text`` space is missing the batch arguments that
    ``Dict.no_op(batch_shape=...)`` / ``Dict.sample(bs)`` pass to every
    sub-space, so ``env.action_space.noop()`` raises and every chat command
    issued through it (the reset-time /gamerule list) is silently dropped.
    Accept and ignore them.
    """

    def no_op(self, batch_shape=()):
        return ""

    def sample(self, bs=None):
        return ""


class ChatAction(Action):
    """A simple text-channel action that maps a string to a chat command.

    Empty string is treated as a no-op.
    """

    def to_string(self) -> str:
        return "chat"

    def xml_template(self) -> str:
        return "<ChatCommands/>"

    def __init__(self) -> None:
        self._command = "chat"
        super().__init__(self._command, _ChatText([1]))
