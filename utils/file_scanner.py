"""File scanner for discovering NetCDF forecast files in realm directories."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import config


class FileScanner:
    """Scans configured realm paths to discover available forecast files."""

    def __init__(self, realm: str = "test_realm"):
        """Initialize the file scanner.

        Args:
            realm: Which realm to scan ("test_realm" or "user_realm")
        """
        self.realm = realm
        self.realm_path = Path(config.REALM_PATHS.get(realm, ""))
        self._cache: Dict = {}

    def set_realm(self, realm: str):
        """Change the realm being scanned."""
        self.realm = realm
        self.realm_path = Path(config.REALM_PATHS.get(realm, ""))
        self._cache = {}

    def get_cycles(self) -> List[str]:
        """Get list of available forecast cycles (e.g., 20250308T1200Z).

        Returns:
            List of cycle directory names, sorted newest first
        """
        if not self.realm_path.exists():
            return []

        cycles = []
        cycle_pattern = re.compile(r"^\d{8}T\d{4}Z$")

        for item in self.realm_path.iterdir():
            if item.is_dir() and cycle_pattern.match(item.name):
                cycles.append(item.name)

        return sorted(cycles, reverse=True)

    def get_steps(self, cycle: str) -> List[str]:
        """Get list of available processing steps for a cycle.

        Args:
            cycle: Cycle name (e.g., "20250308T1200Z")

        Returns:
            List of available step names
        """
        cycle_path = self.realm_path / cycle
        if not cycle_path.exists():
            return []

        steps = []
        for item in cycle_path.iterdir():
            if item.is_dir() and item.name in config.STEPS:
                steps.append(item.name)

        return sorted(steps)

    def get_parameters(self, cycle: str, step: str) -> List[str]:
        """Get list of available parameters for a cycle and step.

        Args:
            cycle: Cycle name
            step: Step name (e.g., "blend")

        Returns:
            List of available parameter names
        """
        param_path = self.realm_path / cycle / step
        if not param_path.exists():
            return []

        parameters = []
        for item in param_path.iterdir():
            if item.is_dir() and item.name in config.PARAMETERS:
                parameters.append(item.name)

        return sorted(parameters)

    def get_outputs(self, cycle: str, step: str, parameter: str) -> List[str]:
        """Get list of available output types for a cycle/step/parameter.

        Args:
            cycle: Cycle name
            step: Step name
            parameter: Parameter name (e.g., "precipacc24h")

        Returns:
            List of available output type names
        """
        output_path = self.realm_path / cycle / step / parameter
        if not output_path.exists():
            return []

        outputs = []
        for item in output_path.iterdir():
            if item.is_dir():
                # Check if directory contains any .nc files
                nc_files = list(item.glob("*.nc"))
                if nc_files:
                    outputs.append(item.name)

        return sorted(outputs)

    def get_files(
        self, cycle: str, step: str, parameter: str, output: str
    ) -> List[Dict]:
        """Get list of NetCDF files for the specified path.

        Args:
            cycle: Cycle name
            step: Step name
            parameter: Parameter name
            output: Output type name

        Returns:
            List of dicts with file info (path, validtime, leadhour)
        """
        file_path = self.realm_path / cycle / step / parameter / output
        if not file_path.exists():
            return []

        files = []
        for nc_file in sorted(file_path.glob("*.nc")):
            file_info = self._parse_filename(nc_file, cycle)
            if file_info:
                files.append(file_info)

        return files

    def _parse_filename(self, filepath: Path, cycle: str) -> Optional[Dict]:
        """Parse IMPROVER filename to extract validtime and leadhour.

        Filename format: YYYYMMDDTHH00Z-PTXXXXHXXM-variable-PTXXH.nc
        Example: 20250309T1500Z-PT0027H00M-precipitation_accumulation-PT24H.nc

        Args:
            filepath: Path to the NetCDF file
            cycle: Cycle name for computing lead hour

        Returns:
            Dict with path, validtime, leadhour, or None if parsing fails
        """
        filename = filepath.name

        # Try to extract validtime from filename
        # Pattern: starts with YYYYMMDDTHH00Z
        validtime_match = re.match(r"^(\d{8}T\d{4}Z)", filename)
        if not validtime_match:
            # Try merged file pattern with lead range: *_XXX_XXX.nc
            lead_range_match = re.search(r"_(\d{3})_(\d{3})\.nc$", filename)
            if lead_range_match:
                return {
                    "path": str(filepath),
                    "filename": filename,
                    "validtime": None,
                    "leadhour_start": int(lead_range_match.group(1)),
                    "leadhour_end": int(lead_range_match.group(2)),
                    "is_merged": True,
                }
            return None

        validtime_str = validtime_match.group(1)

        # Extract lead hour from PTXXXXHXXM pattern
        leadhour_match = re.search(r"-PT(\d{4})H", filename)
        if leadhour_match:
            leadhour = int(leadhour_match.group(1))
        else:
            # Calculate from validtime and cycle
            try:
                vt = datetime.strptime(validtime_str, "%Y%m%dT%H%MZ")
                ct = datetime.strptime(cycle, "%Y%m%dT%H%MZ")
                leadhour = int((vt - ct).total_seconds() / 3600)
            except ValueError:
                leadhour = None

        return {
            "path": str(filepath),
            "filename": filename,
            "validtime": validtime_str,
            "leadhour": leadhour,
            "is_merged": False,
        }

    def get_validtimes(
        self, cycle: str, step: str, parameter: str, output: str
    ) -> List[Tuple[str, int]]:
        """Get list of (validtime, leadhour) tuples for dropdown.

        Args:
            cycle: Cycle name
            step: Step name
            parameter: Parameter name
            output: Output type name

        Returns:
            List of (validtime_str, leadhour) tuples
        """
        files = self.get_files(cycle, step, parameter, output)
        validtimes = []

        for f in files:
            if f.get("is_merged"):
                # For merged files, create entries for the lead hour range
                validtimes.append(
                    (f"Lead {f['leadhour_start']}-{f['leadhour_end']}h", f["leadhour_start"])
                )
            elif f.get("validtime"):
                validtimes.append((f["validtime"], f.get("leadhour", 0)))

        return sorted(validtimes, key=lambda x: x[1])

    def get_file_for_validtime(
        self,
        cycle: str,
        step: str,
        parameter: str,
        output: str,
        validtime: str,
    ) -> Optional[str]:
        """Get the file path for a specific validtime.

        Args:
            cycle: Cycle name
            step: Step name
            parameter: Parameter name
            output: Output type name
            validtime: Valid time string or "Lead XX-XXh" for merged files

        Returns:
            File path string or None
        """
        files = self.get_files(cycle, step, parameter, output)

        for f in files:
            if f.get("is_merged"):
                label = f"Lead {f['leadhour_start']}-{f['leadhour_end']}h"
                if validtime == label:
                    return f["path"]
            elif f.get("validtime") == validtime:
                return f["path"]

        return None

    def refresh(self):
        """Clear the cache to force re-scanning."""
        self._cache = {}

    def get_hierarchy(self) -> Dict:
        """Build complete file hierarchy for the realm.

        Returns:
            Nested dict: {cycle: {step: {parameter: {output: [files]}}}}
        """
        hierarchy = {}

        for cycle in self.get_cycles():
            hierarchy[cycle] = {}
            for step in self.get_steps(cycle):
                hierarchy[cycle][step] = {}
                for param in self.get_parameters(cycle, step):
                    hierarchy[cycle][step][param] = {}
                    for output in self.get_outputs(cycle, step, param):
                        hierarchy[cycle][step][param][output] = self.get_files(
                            cycle, step, param, output
                        )

        return hierarchy
