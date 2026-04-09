"""Core environment, level, and baseline planning utilities.

This module keeps the simulation-side building blocks in one place:
1. Build reusable platform templates and sampled levels.
2. Simulate platform dynamics and classical A* baselines.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import heapq
import math

import numpy as np
import numpy.typing as npt

# Shared type aliases used across environment simulation and model fitting.
GridPosition = tuple[int, int]
PlannerState = tuple[int, int, int, int]
ObservationData = dict[str, Any]
FloatMatrix = npt.NDArray[np.float32]
IntVector = npt.NDArray[np.int32]
TrainingDataset = tuple[FloatMatrix, FloatMatrix, IntVector]
EnemySignature = tuple[int, int]
EnvironmentSignature = tuple[
    GridPosition,
    int,
    int,
    tuple[EnemySignature, ...],
]

# Compact action encoding reused by search, MPC rollout, and disturbance
# models.
ACTION_COMMANDS = {
    0: ("idle", 0, False),
    1: ("left", -1, False),
    2: ("right", 1, False),
    3: ("jump", 0, True),
    4: ("jump_left", -1, True),
    5: ("jump_right", 1, True),
}
AVAILABLE_ACTIONS = tuple(ACTION_COMMANDS)
ACTION_SPACE_SIZE = len(AVAILABLE_ACTIONS)
ACTION_DISTURBANCE_MAP = {
    0: (0, 1, 2),
    1: (0, 1, 4),
    2: (0, 2, 5),
    3: (0, 3, 4, 5),
    4: (0, 1, 3, 4),
    5: (0, 2, 3, 5),
}
JUMP_VELOCITY = 2
GRAVITY_FORCE = 1
MAX_FALL_VELOCITY = 3


def estimate_goal_distance(
    position: GridPosition,
    goal: GridPosition,
) -> float:
    """Computes the heuristic distance used by A* and reward shaping."""
    row, col = position
    goal_row, goal_col = goal
    return abs(col - goal_col) + 1.8 * max(0, row - goal_row) + 1.2 * max(
        0, goal_row - row
    )


def build_segment(row: int, start_col: int, end_col: int) -> set[GridPosition]:
    """Builds one horizontal platform segment on the given row."""
    return {(row, col) for col in range(start_col, end_col + 1)}


def mirror_level_tiles(
    cells: set[GridPosition],
    width: int,
) -> set[GridPosition]:
    """Mirrors a set of occupied cells across the level width."""
    return {(row, width - 1 - col) for row, col in cells}


def mirror_patrol_path(
    path: list[GridPosition],
    width: int,
) -> list[GridPosition]:
    """Mirrors an enemy patrol path across the level width."""
    return [(row, width - 1 - col) for row, col in path]


# =============================================================================
# LevelScenario construction utilities.
# =============================================================================
@dataclass(frozen=True)
class LevelTemplate:
    """Static blueprint for one handcrafted platform layout."""

    name: str
    width: int
    height: int
    solid_tiles: set[GridPosition]
    start: GridPosition
    goal: GridPosition
    enemy_paths: list[list[GridPosition]]


def build_level_templates() -> list[LevelTemplate]:
    """Builds the handcrafted platform templates used in the experiments.

    Returns:
        A list containing the original layouts and mirrored variants so
        the training and evaluation sets cover both traversal directions.
    """
    width, height = 24, 14
    floor_tiles = build_segment(height - 1, 0, width - 1)
    templates = [
        LevelTemplate(
            "staircase",
            width,
            height,
            floor_tiles
            | build_segment(10, 3, 7)
            | build_segment(8, 8, 12)
            | build_segment(6, 13, 17)
            | build_segment(4, 18, 22),
            (12, 1),
            (3, 21),
            [
                [(12, c) for c in range(5, 9)],
                [(9, c) for c in range(4, 7)],
                [(5, c) for c in range(14, 17)],
            ],
        ),
        LevelTemplate(
            "canyon",
            width,
            height,
            floor_tiles
            | build_segment(11, 0, 5)
            | build_segment(11, 9, 14)
            | build_segment(11, 18, 23)
            | build_segment(8, 4, 8)
            | build_segment(8, 12, 16)
            | build_segment(5, 8, 12)
            | build_segment(3, 15, 21),
            (10, 2),
            (2, 19),
            [
                [(10, c) for c in range(10, 14)],
                [(7, c) for c in range(5, 8)],
                [(2, c) for c in range(16, 20)],
            ],
        ),
        LevelTemplate(
            "tower",
            width,
            height,
            floor_tiles
            | build_segment(10, 2, 6)
            | build_segment(9, 9, 12)
            | build_segment(7, 5, 9)
            | build_segment(6, 13, 16)
            | build_segment(4, 10, 14)
            | build_segment(3, 18, 22),
            (12, 2),
            (2, 20),
            [
                [(12, c) for c in range(6, 10)],
                [(8, c) for c in range(10, 12)],
                [(5, c) for c in range(11, 14)],
            ],
        ),
        LevelTemplate(
            "bridge",
            width,
            height,
            floor_tiles
            | build_segment(11, 0, 4)
            | build_segment(11, 7, 11)
            | build_segment(9, 10, 14)
            | build_segment(7, 13, 17)
            | build_segment(5, 17, 22),
            (10, 1),
            (4, 20),
            [
                [(12, c) for c in range(4, 8)],
                [(8, c) for c in range(10, 13)],
                [(4, c) for c in range(18, 21)],
            ],
        ),
        LevelTemplate(
            "gauntlet",
            width,
            height,
            floor_tiles
            | build_segment(11, 0, 3)
            | build_segment(11, 5, 8)
            | build_segment(11, 10, 13)
            | build_segment(11, 15, 18)
            | build_segment(11, 20, 23)
            | build_segment(8, 3, 6)
            | build_segment(8, 9, 12)
            | build_segment(8, 15, 18)
            | build_segment(5, 7, 10)
            | build_segment(5, 13, 16)
            | build_segment(3, 18, 22),
            (10, 1),
            (2, 20),
            [
                [(10, c) for c in range(5, 8)],
                [(7, c) for c in range(9, 12)],
                [(4, c) for c in range(14, 17)],
            ],
        ),
        LevelTemplate(
            "overhang",
            width,
            height,
            floor_tiles
            | build_segment(11, 0, 6)
            | build_segment(9, 5, 10)
            | build_segment(7, 9, 13)
            | build_segment(9, 15, 20)
            | build_segment(5, 17, 22)
            | build_segment(3, 12, 16),
            (10, 1),
            (2, 14),
            [
                [(10, c) for c in range(2, 6)],
                [(8, c) for c in range(16, 19)],
                [(4, c) for c in range(18, 21)],
            ],
        ),
    ]
    return templates + [
        LevelTemplate(
            template.name + "-mirror",
            template.width,
            template.height,
            mirror_level_tiles(template.solid_tiles, template.width),
            (template.start[0], template.width - 1 - template.start[1]),
            (template.goal[0], template.width - 1 - template.goal[1]),
            [mirror_patrol_path(path, template.width)
             for path in template.enemy_paths],
        )
        for template in templates
    ]


LEVEL_TEMPLATES = build_level_templates()


def level_family_name(name: str) -> str:
    """Normalizes a level name to its underlying template family."""
    n = str(name).split("-enemy")[0]
    return n[:-7] if n.endswith("-mirror") else n


LEVEL_FAMILY_NAMES = tuple(
    dict.fromkeys(level_family_name(t.name) for t in LEVEL_TEMPLATES)
)
LEVEL_FAMILY_INDEX = {n: i for i, n in enumerate(LEVEL_FAMILY_NAMES)}


def build_level_scenario(
    template: LevelTemplate, rng: np.random.Generator
) -> "LevelScenario":
    """Samples a runnable level from a static template.

    Args:
        template: Template that defines geometry, start/goal, and patrol
            routes.
        rng: Random generator used to randomize initial enemy phases and
            directions.

    Returns:
        A fresh level instance ready to pass into the environment.
    """
    return LevelScenario(
        template.name,
        template.width,
        template.height,
        set(template.solid_tiles),
        template.start,
        template.goal,
        [
            PatrolEnemy(
                path[:],
                int(rng.integers(0, len(path))),
                1 if int(rng.integers(0, 2)) else -1,
            )
            for path in template.enemy_paths
        ],
    )


def is_solid_tile(
    solid_tiles: set[GridPosition], row: int, col: int, height: int, width: int
) -> bool:
    """Checks whether a cell is outside bounds or occupied by solid terrain."""
    return (
        col < 0
        or col >= width
        or row < 0
        or row >= height
        or (row, col) in solid_tiles
    )


def is_grounded_state(
        position: GridPosition,
        solid_tiles: set[GridPosition],
        width: int,
        height: int) -> bool:
    """Checks whether the agent is standing on solid ground."""
    return is_solid_tile(
        solid_tiles,
        position[0] + 1,
        position[1],
        height,
        width)


def nearest_patrol_distance(
    position: GridPosition,
    enemy_positions: set[GridPosition] | frozenset[GridPosition],
) -> float:
    """Returns the Manhattan distance from the agent to the nearest enemy."""
    if not enemy_positions:
        return 999.0
    row, col = position
    return float(
        min(
            abs(row - enemy_row) + abs(col - enemy_col)
            for enemy_row, enemy_col in enemy_positions
        )
    )


@dataclass
class PatrolEnemy:
    """Enemy that moves back and forth along a fixed patrol path."""

    path: list[GridPosition]
    index: int = 0
    direction: int = 1

    def current_position(self) -> GridPosition:
        """Returns the patrol cell currently occupied by the enemy."""
        return self.path[self.index]

    def clone(self) -> "PatrolEnemy":
        """Creates an advanceable copy without mutating the original."""
        return PatrolEnemy(self.path[:], self.index, self.direction)

    def period(self) -> int:
        """Returns the full back-and-forth patrol cycle length."""
        return max(1, 2 * len(self.path) - 2)

    def advance(
        self,
        blocked_cells: Collection[GridPosition] = (),
    ) -> None:
        """Advances the patrol by one step while respecting blocked cells."""
        if len(self.path) < 2:
            return
        next_index = self.index + self.direction
        if next_index < 0 or next_index >= len(self.path):
            self.direction *= -1
            next_index = self.index + self.direction
        if self.path[next_index] in blocked_cells:
            self.direction *= -1
            next_index = self.index + self.direction
            if (
                next_index < 0
                or next_index >= len(self.path)
                or self.path[next_index] in blocked_cells
            ):
                return
        self.index = next_index


# =============================================================================
# Enemy patrol helpers.
# =============================================================================
def patrol_signature(enemy: PatrolEnemy) -> EnemySignature:
    """Builds a compact signature for one enemy's patrol phase."""
    return enemy.index, enemy.direction


def patrol_group_signature(
    enemies: Sequence[PatrolEnemy],
) -> tuple[EnemySignature, ...]:
    """Builds a compact signature for the full enemy group state."""
    return tuple(patrol_signature(enemy) for enemy in enemies)


def patrol_timetable_key(
    enemies: Sequence[PatrolEnemy],
) -> tuple[tuple[tuple[GridPosition, ...], int, int], ...]:
    """Creates the cache key for a patrol group's repeating timetable."""
    return tuple(
        (tuple(enemy.path), enemy.index, enemy.direction) for enemy in enemies
    )


_PATROL_TIMETABLE_CACHE = {}


def advance_patrol_group(enemies: Sequence[PatrolEnemy]) -> None:
    """Advances all enemies by one step while preventing direct overlaps."""
    for enemy in enemies:
        enemy.advance(
            {
                other_enemy.current_position()
                for other_enemy in enemies
                if other_enemy is not enemy
            }
        )


def patrol_cycle_length(enemies: Sequence[PatrolEnemy]) -> int:
    """Computes the least common multiple of enemy patrol periods."""
    cycle_length = 1
    for enemy in enemies:
        cycle_length = math.lcm(cycle_length, enemy.period())
    return cycle_length


def patrol_positions_after_steps(
    enemies: Sequence[PatrolEnemy],
    steps_ahead: int,
) -> set[GridPosition]:
    """Returns occupied enemy cells after simulating `steps_ahead` steps."""
    future_enemies = [enemy.clone() for enemy in enemies]
    for _ in range(steps_ahead):
        advance_patrol_group(future_enemies)
    return {enemy.current_position() for enemy in future_enemies}


def patrol_positions_timetable(
    enemies: list[PatrolEnemy],
) -> tuple[frozenset[GridPosition], ...]:
    """Caches the repeating enemy occupancy pattern for predictive planning.

    The enemy patrols are deterministic and periodic, so the planner can reason
    about future collisions by unrolling one full cycle once and reusing it.

    Args:
        enemies: Active enemies in the current level snapshot.

    Returns:
        A tuple whose elements are occupied enemy cells at each future
        phase.
    """
    if not enemies:
        return (frozenset(),)
    cache_key = patrol_timetable_key(enemies)
    table = _PATROL_TIMETABLE_CACHE.get(cache_key)
    if table is not None:
        return table
    simulated_enemies = [enemy.clone() for enemy in enemies]
    occupancy_table = []
    for _ in range(patrol_cycle_length(simulated_enemies)):
        occupancy_table.append(
            frozenset(enemy.current_position() for enemy in simulated_enemies)
        )
        advance_patrol_group(simulated_enemies)
    _PATROL_TIMETABLE_CACHE[cache_key] = table = tuple(occupancy_table)
    return table


@dataclass
class LevelScenario:
    """Concrete level instance with geometry, endpoints, and live enemies."""

    name: str
    width: int
    height: int
    solid_tiles: set[GridPosition]
    start: GridPosition
    goal: GridPosition
    enemies: list[PatrolEnemy]

    def clone_enemies(self) -> list[PatrolEnemy]:
        """Clones all live enemies for a fresh environment instance."""
        return [enemy.clone() for enemy in self.enemies]

    def with_enemy_limit(self, enemy_limit: int) -> "LevelScenario":
        """Builds a level variant with only the first `enemy_limit` enemies."""
        return LevelScenario(
            f"{self.name}-enemy{enemy_limit}",
            self.width,
            self.height,
            set(self.solid_tiles),
            self.start,
            self.goal,
            [enemy.clone() for enemy in self.enemies[:enemy_limit]],
        )


@dataclass
class StepResult:
    """Reward, termination flag, and diagnostics from one environment step."""

    reward: float
    done: bool
    info: dict[str, float]


def sample_level_scenario(
    rng: np.random.Generator,
    template_indices: list[int] | tuple[int, ...] | None = None,
) -> LevelScenario:
    """Generates one level sampled from the allowed template set."""
    template_id_pool = (
        np.arange(len(LEVEL_TEMPLATES))
        if template_indices is None
        else np.array(list(template_indices), int)
    )
    template = LEVEL_TEMPLATES[
        int(template_id_pool[int(rng.integers(0, len(template_id_pool)))])
    ]
    return build_level_scenario(template, rng)


def simulate_agent_transition(
    level: LevelScenario,
    position: GridPosition,
    velocity_y: int,
    action: int,
    blocked_cells: set[GridPosition] | frozenset[GridPosition],
) -> tuple[GridPosition | None, int, dict[str, float]]:
    """Applies one platform-action transition under collision and hazard rules.

    Args:
        level: LevelScenario that provides static geometry.
        position: Current agent grid position.
        velocity_y: Current vertical velocity.
        action: Encoded action index.
        blocked_cells: Cells that should be treated as instant-failure
            hazards.

    Returns:
        A tuple of next position, next vertical velocity, and diagnostic
        flags. The position is `None` when the agent collides with a
        hazard.
    """
    _, dx, jump_requested = ACTION_COMMANDS[action]
    row, col = position
    info = {"horizontal_blocked": 0.0, "hazard_collision": 0.0}
    # Resolve jump intent before horizontal movement so the agent can
    # climb from grounded states in the same action.
    if jump_requested and is_grounded_state(
        position, level.solid_tiles, level.width, level.height
    ):
        velocity_y = -JUMP_VELOCITY
    if dx:
        next_col = col + dx
        if is_solid_tile(
            level.solid_tiles, row, next_col, level.height, level.width
        ):
            info["horizontal_blocked"] = 1.0
        elif (row, next_col) in blocked_cells:
            info["hazard_collision"] = 1.0
            return None, velocity_y, info
        else:
            col = next_col
    # Vertical motion is integrated one cell at a time so ceilings, floors, and
    # hazards are all handled consistently.
    if velocity_y:
        vertical_step = -1 if velocity_y < 0 else 1
        for _ in range(abs(velocity_y)):
            next_row = row + vertical_step
            if is_solid_tile(
                level.solid_tiles, next_row, col, level.height, level.width
            ):
                velocity_y = 0
                break
            row = next_row
            if (row, col) in blocked_cells:
                info["hazard_collision"] = 1.0
                return None, velocity_y, info
    velocity_y = (
        0
        if is_grounded_state(
            (row, col), level.solid_tiles, level.width, level.height
        )
        else min(velocity_y + GRAVITY_FORCE, MAX_FALL_VELOCITY)
    )
    return (row, col), velocity_y, info


def plan_astar_actions(
    level: LevelScenario,
    start_state: tuple[int, int, int],
    blocked_dynamic: set[GridPosition] | frozenset[GridPosition],
    enemy_states: list[PatrolEnemy] | None = None,
    use_predictive_hazards: bool = False,
    max_expansions: int = 12000,
) -> list[int] | None:
    """Plans a shortest-feasible action sequence with hazard lookahead.

    Args:
        level: LevelScenario snapshot to plan on.
        start_state: `(row, col, vertical_velocity)` of the agent.
        blocked_dynamic: Hazard cells at the current time step.
        enemy_states: Live enemy objects used when predictive hazards
            are enabled.
        use_predictive_hazards: Whether to expand search through the
            full enemy cycle.
        max_expansions: Hard cap on A* node expansions.

    Returns:
        A list of action ids that reaches the goal, or `None` when no
        plan is found.
    """
    table: tuple[frozenset[GridPosition], ...] = (
        patrol_positions_timetable(enemy_states)
        if use_predictive_hazards and enemy_states
        else (frozenset(blocked_dynamic),)
    )
    if (start_state[0], start_state[1]) in table[0]:
        return None

    # Add the enemy phase to the search state so A* can avoid future
    # collisions.
    cyc = len(table)
    start: PlannerState = (start_state[0], start_state[1], start_state[2], 0)
    pq: list[tuple[float, float, PlannerState]] = [
        (
            estimate_goal_distance(
                (start_state[0], start_state[1]), level.goal),
            0.0,
            start,
        )
    ]
    cost: dict[PlannerState, float] = {start: 0.0}
    parent: dict[PlannerState, tuple[PlannerState, int]] = {}
    seen: set[PlannerState] = set()
    n = 0

    while pq and n < max_expansions:
        _, g, s = heapq.heappop(pq)
        if s in seen:
            continue
        seen.add(s)
        n += 1
        r, c, vy, p = s

        if (r, c) == level.goal:
            action_plan = []
            while s in parent:
                s, a = parent[s]
                action_plan.append(a)
            return action_plan[::-1]

        # Expand legal actions and reject transitions that collide now
        # or one step later with the predicted enemy occupancy.
        for a in AVAILABLE_ACTIONS:
            pos, nvy, _ = simulate_agent_transition(
                level, (r, c), vy, a, table[p])
            if pos is None or pos in table[(p + 1) % cyc]:
                continue
            t: PlannerState = (pos[0], pos[1], nvy, (p + 1) % cyc)
            ng = g + 1.0
            if ng < cost.get(t, 1e9):
                cost[t] = ng
                parent[t] = (s, a)
                heapq.heappush(
                    pq,
                    (ng + estimate_goal_distance(pos, level.goal), ng, t),
                )
    return None


def fallback_action_choice(observation: ObservationData) -> int:
    """Selects a simple heuristic action when planning fails."""
    r, c = observation["position"]
    gr, gc = observation["goal"]
    if observation["grounded"] and gr < r:
        return 5 if gc > c else 4 if gc < c else 3
    return 2 if gc > c else 1 if gc < c else 0


def snapshot_level_from_observation(
    observation: ObservationData,
) -> LevelScenario:
    """Builds a static level snapshot from one controller observation."""
    return LevelScenario(
        "snapshot",
        int(observation["width"]),
        int(observation["height"]),
        set(observation["solid_tiles"]),
        observation["position"],
        observation["goal"],
        [],
    )


def first_planned_action(
    plan: Sequence[int] | None,
    observation: ObservationData,
) -> int:
    """Returns the first planned action or the heuristic fallback action."""
    return plan[0] if plan else fallback_action_choice(observation)


# =============================================================================
# Environment and baseline controllers.
# =============================================================================
class PlatformerEnv:
    """Grid platformer environment with deterministic enemy patrol dynamics."""

    def __init__(
        self,
        level: LevelScenario,
        max_steps: int = 120,
    ) -> None:
        self.level = level
        self.max_steps = max_steps
        self.agent_position = level.start
        self.velocity_y = 0
        self.enemies = level.clone_enemies()
        self.steps = 0
        self.blocked_actions = 0
        self.path_length = 0
        self.done = False
        self.success = False
        self.total_reward = 0.0
        self.recent_positions = [level.start]
        self._enemy_positions_cache = None
        self._future_enemy_cache = {}
        self._state_signature_cache = None
        self._observation_cache = None

    def _clear_cache(self) -> None:
        """Clears cached observations and enemy-position summaries."""
        self._enemy_positions_cache = None
        self._future_enemy_cache = {}
        self._state_signature_cache = None
        self._observation_cache = None

    def enemy_positions(self) -> set[GridPosition]:
        """Returns the current set of occupied enemy cells."""
        if self._enemy_positions_cache is None:
            self._enemy_positions_cache = {
                enemy.current_position() for enemy in self.enemies
            }
        return self._enemy_positions_cache

    def future_enemy_positions(self, steps_ahead: int) -> set[GridPosition]:
        """Returns occupied enemy cells after `steps_ahead` future steps."""
        if steps_ahead not in self._future_enemy_cache:
            self._future_enemy_cache[steps_ahead] = (
                patrol_positions_after_steps(self.enemies, steps_ahead)
            )
        return self._future_enemy_cache[steps_ahead]

    def state_signature(self) -> EnvironmentSignature:
        """Returns a hashable snapshot of the environment planning state."""
        if self._state_signature_cache is None:
            self._state_signature_cache = (
                self.agent_position,
                self.velocity_y,
                self.steps,
                patrol_group_signature(self.enemies),
            )
        return self._state_signature_cache

    def observation(self) -> ObservationData:
        """Builds the controller-facing observation dictionary."""
        if self._observation_cache is None:
            self._observation_cache = {
                "position": self.agent_position,
                "goal": self.level.goal,
                "velocity_y": self.velocity_y,
                "grounded": int(
                    is_grounded_state(
                        self.agent_position,
                        self.level.solid_tiles,
                        self.level.width,
                        self.level.height,
                    )
                ),
                "solid_tiles": self.level.solid_tiles,
                "enemy_positions": tuple(sorted(self.enemy_positions())),
                "width": self.level.width,
                "height": self.level.height,
                "estimate_goal_distance": estimate_goal_distance(
                    self.agent_position, self.level.goal
                ),
                "enemy_count": len(self.enemies),
            }
        return self._observation_cache

    def clone(self) -> "PlatformerEnv":
        """Creates a simulation copy for rollout planning."""
        cloned_env = PlatformerEnv(self.level, self.max_steps)
        cloned_env.agent_position = self.agent_position
        cloned_env.velocity_y = self.velocity_y
        cloned_env.enemies = [enemy.clone() for enemy in self.enemies]
        cloned_env.steps = self.steps
        cloned_env.blocked_actions = self.blocked_actions
        cloned_env.path_length = self.path_length
        cloned_env.done = self.done
        cloned_env.success = self.success
        cloned_env.total_reward = self.total_reward
        cloned_env.recent_positions = self.recent_positions[:]
        return cloned_env

    def step(self, action: int) -> StepResult:
        """Runs one environment step and computes the shaped reward signal.

        The reward blends sparse task success with shaping terms for progress,
        oscillation, blocked moves, and enemy proximity so both A*
        rollouts and learned models see the same training signal.
        """
        self.steps += 1
        old = estimate_goal_distance(self.agent_position, self.level.goal)
        pos, vy, info = simulate_agent_transition(
            self.level,
            self.agent_position,
            self.velocity_y,
            action,
            self.enemy_positions(),
        )
        reward = -0.25
        hazard_flag = 0.0
        if pos is None:
            reward -= 18.0
            hazard_flag = 1.0
            self.done = True
        else:
            moved = pos != self.agent_position
            osc = (
                moved
                and len(self.recent_positions) > 1
                and pos == self.recent_positions[-2]
                and self.recent_positions[-1] != self.recent_positions[-2]
            )
            if info["horizontal_blocked"]:
                self.blocked_actions += 1
                reward -= 0.35
            if moved:
                self.path_length += 1
            elif action:
                reward -= 0.15
            if osc:
                reward -= 0.6
            self.agent_position = pos
            self.velocity_y = vy
            self.recent_positions = (self.recent_positions + [pos])[-5:]
            self._clear_cache()
            reward += 0.3 * np.clip(
                old - estimate_goal_distance(pos, self.level.goal), -3.0, 3.0
            )
            if pos == self.level.goal:
                reward += 25.0
                self.done = True
                self.success = True
        if not self.done:
            advance_patrol_group(self.enemies)
            self._clear_cache()
            if self.agent_position in self.enemy_positions():
                reward -= 18.0
                hazard_flag = 1.0
                self.done = True
            else:
                d = nearest_patrol_distance(
                    self.agent_position, self.enemy_positions())
                if d <= 2.0:
                    reward -= 0.22 * (3.0 - d)
        if self.steps >= self.max_steps:
            self.done = True
        self.total_reward += reward
        return StepResult(
            reward,
            self.done,
            {
                "hazard_collision": hazard_flag,
                "horizontal_blocked": (
                    0.0 if pos is None else float(info["horizontal_blocked"])
                ),
            },
        )


class StaticAStarController:
    """Baseline controller that plans against static terrain only."""

    name = "Static A*"

    @staticmethod
    def select_action(
        observation: ObservationData, _env: PlatformerEnv | None = None
    ) -> int:
        """Selects the next action from a terrain-only A* plan."""
        level = snapshot_level_from_observation(observation)
        p = observation["position"]
        plan = plan_astar_actions(
            level,
            (p[0], p[1], int(observation["velocity_y"])),
            set(),
        )
        return first_planned_action(plan, observation)


class DynamicAStarController:
    """Baseline controller that replans with current or predicted hazards."""

    name = "Dynamic A*"

    def __init__(self, use_predictive_hazards: bool = True) -> None:
        self.use_predictive_hazards = use_predictive_hazards

    def select_action(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> int:
        """Selects the next action from a hazard-aware A* plan."""
        level = snapshot_level_from_observation(observation)
        p = observation["position"]
        plan = plan_astar_actions(
            level,
            (p[0], p[1], int(observation["velocity_y"])),
            set(observation["enemy_positions"]),
            None if environment is None else environment.enemies,
            self.use_predictive_hazards and environment is not None,
        )
        return first_planned_action(plan, observation)
