"""Hard-constraint configuration — dataset time windows must not be modified."""
from __future__ import annotations

from dataclasses import dataclass

from src.hanif.utils.logging_utils import get_logger

LOGGER = get_logger()


@dataclass
class ConstraintConfig:
    time_windows_mode: str = "hard_original"
    allow_late_service: bool = False
    relax_time_windows: bool = False
    modify_instance_time_windows: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "ConstraintConfig":
        raw = dict(config.get("constraints") or {})
        # Allow the same keys under environment (RC-only / experiment configs).
        env = config.get("environment") or {}
        for key in (
            "time_windows_mode",
            "allow_late_service",
            "relax_time_windows",
            "modify_instance_time_windows",
        ):
            if key in env and key not in raw:
                raw[key] = env[key]
        return cls(
            time_windows_mode=str(raw.get("time_windows_mode", "hard_original")),
            allow_late_service=bool(raw.get("allow_late_service", False)),
            relax_time_windows=bool(raw.get("relax_time_windows", False)),
            modify_instance_time_windows=bool(raw.get("modify_instance_time_windows", False)),
        )

    def validate_or_fail(self) -> None:
        if self.modify_instance_time_windows:
            raise ValueError(
                "modify_instance_time_windows=true is forbidden — Schneider instances must stay unchanged."
            )
        if self.relax_time_windows:
            raise ValueError(
                "relax_time_windows=true is forbidden — use original hard time windows only."
            )
        if self.time_windows_mode == "hard_original":
            if self.allow_late_service:
                raise ValueError(
                    "allow_late_service=true conflicts with time_windows_mode=hard_original."
                )
            if self.relax_time_windows or self.modify_instance_time_windows:
                raise ValueError("Hard original time windows require all relaxation flags disabled.")

    @property
    def hard_time_windows(self) -> bool:
        return (
            self.time_windows_mode == "hard_original"
            and not self.allow_late_service
            and not self.relax_time_windows
        )


def log_constraint_startup(cfg: ConstraintConfig) -> None:
    cfg.validate_or_fail()
    LOGGER.info(
        "Using original Schneider RC _21 time windows. No relaxation. No dataset modification. "
        "mode=%s allow_late=%s",
        cfg.time_windows_mode,
        cfg.allow_late_service,
    )
